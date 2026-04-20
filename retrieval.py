import chromadb
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re

CHROMA_PATH  = "./chroma_db"
BM25_PATH    = "./bm25_index.pkl"
EMB_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Load resources (call once) ────────────────────────────────────────────────
def load_resources():
    embedder = SentenceTransformer(EMB_MODEL, device="cuda")

    client      = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = {
        "linkedin_jobs"       : client.get_collection("linkedin_jobs"),
        "glassdoor_reviews"   : client.get_collection("glassdoor_reviews"),
        "stackoverflow_survey": client.get_collection("stackoverflow_survey"),
    }

    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    reranker = CrossEncoder(RERANK_MODEL, device="cuda")

    return embedder, collections, bm25_data, reranker

# ── Tokenize helper ───────────────────────────────────────────────────────────
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return text.split()

# ── Dense retrieval ───────────────────────────────────────────────────────────
def dense_retrieve(query, collections, embedder, top_k=50):
    vec     = embedder.encode([query], normalize_embeddings=True).tolist()
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
                "id"    : f"{name}_{doc[:20]}",
                "text"  : doc,
                "meta"  : meta,
                "source": name,
                "dense_score": 1 - dist,
            })
    return results

# ── Sparse retrieval (BM25) ───────────────────────────────────────────────────
def sparse_retrieve(query, bm25_data, top_k=50):
    bm25   = bm25_data["bm25"]
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idx= np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if scores[idx] == 0:
            continue
        results.append({
            "id"          : bm25_data["ids"][idx],
            "text"        : bm25_data["docs"][idx],
            "meta"        : bm25_data["metas"][idx],
            "source"      : bm25_data["metas"][idx].get("source", "unknown"),
            "sparse_score": float(scores[idx]),
        })
    return results

# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=20):
    scores = {}
    docs   = {}

    for rank, item in enumerate(dense_results):
        uid = item["id"]
        scores[uid] = scores.get(uid, 0) + 1 / (rank + k)
        docs[uid]   = item

    for rank, item in enumerate(sparse_results):
        uid = item["id"]
        scores[uid] = scores.get(uid, 0) + 1 / (rank + k)
        if uid not in docs:
            docs[uid] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {**docs[uid], "rrf_score": round(score, 5)}
        for uid, score in ranked
    ]

# ── Reranking ─────────────────────────────────────────────────────────────────
def rerank(query, candidates, reranker, top_n=5):
    pairs  = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for i, score in enumerate(scores):
        candidates[i]["rerank_score"] = float(score)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

# ── Full pipeline ─────────────────────────────────────────────────────────────
def retrieve_full(query, collections, embedder, bm25_data, reranker,
                  use_rerank=True, top_k=50, final_k=5):
    dense   = dense_retrieve(query, collections, embedder, top_k)
    sparse  = sparse_retrieve(query, bm25_data, top_k)
    fused   = reciprocal_rank_fusion(dense, sparse, top_n=top_k)

    if use_rerank:
        final = rerank(query, fused, reranker, top_n=final_k)
    else:
        final = fused[:final_k]

    return final, {
        "dense_count" : len(dense),
        "sparse_count": len(sparse),
        "fused_count" : len(fused),
        "final_count" : len(final),
    }