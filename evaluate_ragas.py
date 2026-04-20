import os
import json
import requests
import pickle
import numpy as np
import re
import chromadb
from datasets import Dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH  = "./chroma_db"
BM25_PATH    = "./bm25_index.pkl"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_BASE  = "http://localhost:11434"
MODEL_NAME   = "qwen3.5:latest"
EMB_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Test questions ────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    "What skills do machine learning engineers need at Google?",
    "What is the average salary for a Python backend engineer?",
    "Which companies have the best work life balance for software engineers?",
    "What programming languages are most in demand for data engineers?",
    "What do employees say about working at Amazon?",
    "How much do full stack developers earn in the United States?",
    "What are the pros and cons of working at startups vs big tech?",
    "What cloud platforms do backend engineers use most?",
    "Is Rust in demand for systems programming jobs?",
    "What benefits do top tech companies typically offer?",
]

# ── Load resources ────────────────────────────────────────────────────────────
print("Loading embedder...")
embedder = SentenceTransformer(EMB_MODEL, device="cuda")

print("Loading ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collections = {
    "linkedin_jobs"       : client.get_collection("linkedin_jobs"),
    "glassdoor_reviews"   : client.get_collection("glassdoor_reviews"),
    "stackoverflow_survey": client.get_collection("stackoverflow_survey"),
}

print("Loading BM25 index...")
with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)

print("Loading reranker...")
reranker = CrossEncoder(RERANK_MODEL, device="cuda")

# ── Retrieval helpers ─────────────────────────────────────────────────────────
def tokenize(text):
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).split()

def dense_retrieve(query, top_k=30):
    vec = embedder.encode([query], normalize_embeddings=True).tolist()
    results = []
    for name, col in collections.items():
        r = col.query(
            query_embeddings=vec,
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"]
        )
        for doc, meta, dist in zip(
            r["documents"][0], r["metadatas"][0], r["distances"][0]
        ):
            results.append({
                "id": f"{name}::{doc[:30]}",
                "text": doc, "meta": meta,
                "source": name, "dense_score": 1 - dist,
            })
    return results

def sparse_retrieve(query, top_k=30):
    scores  = bm25_data["bm25"].get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "id"    : bm25_data["ids"][i],
            "text"  : bm25_data["docs"][i],
            "meta"  : bm25_data["metas"][i],
            "source": bm25_data["metas"][i].get("source", "unknown"),
            "sparse_score": float(scores[i]),
        }
        for i in top_idx if scores[i] > 0
    ]

def rrf_merge(dense, sparse, k=60, top_n=20):
    scores, docs = {}, {}
    for rank, item in enumerate(dense):
        uid = item["id"]
        scores[uid] = scores.get(uid, 0) + 1 / (rank + k)
        docs[uid] = item
    for rank, item in enumerate(sparse):
        uid = item["id"]
        scores[uid] = scores.get(uid, 0) + 1 / (rank + k)
        if uid not in docs:
            docs[uid] = item
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{**docs[uid], "rrf_score": round(s, 5)} for uid, s in ranked]

def rerank_chunks(query, candidates, top_n=5):
    scores = reranker.predict([[query, c["text"]] for c in candidates])
    for i, s in enumerate(scores):
        candidates[i]["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

def retrieve(query):
    dense  = dense_retrieve(query, top_k=30)
    sparse = sparse_retrieve(query, top_k=30)
    fused  = rrf_merge(dense, sparse, top_n=20)
    return rerank_chunks(query, fused, top_n=5)

# ── Generate answers via Ollama ───────────────────────────────────────────────
def ask_ollama(query, chunks):
    context = "\n\n".join([
        f"[{c['source'].replace('_',' ').title()}]\n{c['text']}"
        for c in chunks
    ])
    prompt = f"""You are a career intelligence assistant.
Answer using ONLY the context below. Be specific and concise.

Context:
{context}

Question: {query}
Answer:"""

    response = requests.post(OLLAMA_URL, json={
        "model" : MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    })
    return response.json().get("response", "")

# ── Run all queries ───────────────────────────────────────────────────────────
print("\nRunning evaluation queries...")
questions, answers, contexts = [], [], []

for i, q in enumerate(TEST_QUESTIONS):
    print(f"  [{i+1}/{len(TEST_QUESTIONS)}] {q[:65]}...")
    chunks  = retrieve(q)
    answer  = ask_ollama(q, chunks)
    context = [c["text"] for c in chunks]

    questions.append(q)
    answers.append(answer)
    contexts.append(context)

    print(f"         Answer: {answer[:100]}...")

# ── Configure RAGAS to use local Qwen via Ollama ──────────────────────────────
print("\nConfiguring RAGAS with local Qwen...")
local_llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE,
    temperature=0,
)
local_emb = OllamaEmbeddings(
    model="llama3",          # use llama3 for embeddings (lighter)
    base_url=OLLAMA_BASE,
)

# ── Run RAGAS ─────────────────────────────────────────────────────────────────
print("Running RAGAS evaluation...")
dataset = Dataset.from_dict({
    "question": questions,
    "answer"  : answers,
    "contexts": contexts,
})

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=local_llm,
    embeddings=local_emb,
)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)
df = results.to_pandas()
print(df[["question", "faithfulness",
          "answer_relevancy", "context_precision"]].to_string())

print("\n── Averages ──")
print(f"Faithfulness:      {df['faithfulness'].mean():.3f}")
print(f"Answer relevancy:  {df['answer_relevancy'].mean():.3f}")
print(f"Context precision: {df['context_precision'].mean():.3f}")

df.to_csv("ragas_results.csv", index=False)
print("\nFull results saved to ragas_results.csv")