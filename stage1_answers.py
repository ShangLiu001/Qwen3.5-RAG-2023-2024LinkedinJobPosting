import json, pickle, numpy as np, re, requests, chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

CHROMA_PATH  = "./chroma_db"
BM25_PATH    = "./bm25_index.pkl"
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "qwen3.5:latest"

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

def tokenize(text):
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).split()

print("Loading models...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
client   = chromadb.PersistentClient(path=CHROMA_PATH)
cols     = {n: client.get_collection(n) for n in
            ["linkedin_jobs","glassdoor_reviews","stackoverflow_survey"]}
with open(BM25_PATH,"rb") as f: bm25_data = pickle.load(f)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")

def retrieve(query):
    vec = embedder.encode([query], normalize_embeddings=True).tolist()
    dense = []
    for name, col in cols.items():
        r = col.query(query_embeddings=vec, n_results=30,
                      include=["documents","metadatas","distances"])
        for doc,meta,dist in zip(r["documents"][0],r["metadatas"][0],r["distances"][0]):
            dense.append({"id":f"{name}::{doc[:30]}","text":doc,"meta":meta,
                          "source":name,"dense_score":1-dist})
    scores  = bm25_data["bm25"].get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:30]
    sparse  = [{"id":bm25_data["ids"][i],"text":bm25_data["docs"][i],
                "meta":bm25_data["metas"][i],"source":bm25_data["metas"][i].get("source","?"),
                "sparse_score":float(scores[i])} for i in top_idx if scores[i]>0]
    sc, docs = {}, {}
    for rank,item in enumerate(dense):
        sc[item["id"]] = sc.get(item["id"],0)+1/(rank+60); docs[item["id"]]=item
    for rank,item in enumerate(sparse):
        sc[item["id"]] = sc.get(item["id"],0)+1/(rank+60)
        if item["id"] not in docs: docs[item["id"]]=item
    fused = [{**docs[uid],"rrf_score":round(s,5)}
             for uid,s in sorted(sc.items(),key=lambda x:x[1],reverse=True)[:20]]
    pair_scores = reranker.predict([[query,c["text"]] for c in fused])
    for i,s in enumerate(pair_scores): fused[i]["rerank_score"]=float(s)
    return sorted(fused,key=lambda x:x["rerank_score"],reverse=True)[:5]

def ask(query, chunks):
    ctx = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in chunks])
    r   = requests.post(OLLAMA_URL, json={
        "model":MODEL_NAME,
        "prompt":f"Answer using ONLY this context.\n\nContext:\n{ctx}\n\nQuestion:{query}\nAnswer:",
        "stream":False})
    return r.json().get("response","")

results = []
for i,q in enumerate(TEST_QUESTIONS):
    print(f"[{i+1}/10] {q[:60]}...")
    chunks  = retrieve(q)
    answer  = ask(q, chunks)
    results.append({
        "question": q,
        "answer"  : answer,
        "contexts": [c["text"] for c in chunks]
    })
    print(f"       {answer[:80]}...")

with open("ragas_input.json","w") as f:
    json.dump(results, f, indent=2)
print("\nSaved ragas_input.json — run stage2_ragas.py next")
