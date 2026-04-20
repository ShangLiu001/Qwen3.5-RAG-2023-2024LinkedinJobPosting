import streamlit as st
import chromadb
import pickle
import numpy as np
import re
import requests
import json
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH  = "./chroma_db"
BM25_PATH    = "./bm25_index.pkl"
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "qwen3.5:latest"
EMB_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    embedder = SentenceTransformer(EMB_MODEL, device="cuda")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = {
        "linkedin_jobs"       : client.get_collection("linkedin_jobs"),
        "glassdoor_reviews"   : client.get_collection("glassdoor_reviews"),
        "stackoverflow_survey": client.get_collection("stackoverflow_survey"),
    }

    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    reranker = CrossEncoder(RERANK_MODEL, device="cuda")
    return embedder, collections, bm25_data, reranker

# ── Helpers ───────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).split()

def dense_retrieve(query, collections, embedder, top_k=50):
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

def sparse_retrieve(query, bm25_data, top_k=50):
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

def rerank(query, candidates, reranker, top_n=5):
    scores = reranker.predict([[query, c["text"]] for c in candidates])
    for i, s in enumerate(scores):
        candidates[i]["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

def retrieve(query, collections, embedder, bm25_data, reranker,
             use_rerank=True, top_k=50, final_k=5):
    dense  = dense_retrieve(query, collections, embedder, top_k)
    sparse = sparse_retrieve(query, bm25_data, top_k)
    fused  = rrf_merge(dense, sparse, top_n=top_k)
    final  = rerank(query, fused, reranker, top_n=final_k) if use_rerank else fused[:final_k]
    return final, {
        "dense": len(dense), "sparse": len(sparse),
        "fused": len(fused), "final": len(final)
    }

def build_prompt(query, chunks):
    context = "\n\n".join([
        f"[{c['source'].replace('_',' ').title()}]\n{c['text']}"
        for c in chunks
    ])
    return f"""You are a career intelligence assistant.
Answer using ONLY the context below. Be specific and cite sources.
If context is insufficient, say so.

Context:
{context}

Question: {query}
Answer:"""

def stream_ollama(prompt):
    with requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME, "prompt": prompt, "stream": True
    }, stream=True) as r:
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("response", "")
                if data.get("done"):
                    break

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareerRAG",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💼 CareerRAG")
    st.markdown("*Career intelligence from 625k real data points*")
    st.divider()

    st.markdown("**SOURCES**")
    use_linkedin      = st.checkbox("LinkedIn Jobs",         value=True)
    use_glassdoor     = st.checkbox("Glassdoor Reviews",     value=True)
    use_stackoverflow = st.checkbox("Stack Overflow Survey", value=True)

    st.divider()
    st.markdown("**RETRIEVAL**")
    use_rerank = st.toggle("Reranking (BGE)", value=True)
    top_k      = st.slider("Final results", 1, 10, 5)

    st.divider()
    st.markdown("**TRY ASKING**")
    examples = [
        "What skills do ML engineers need?",
        "What's it like at Google as a SWE?",
        "How much do Python engineers earn?",
        "Which companies have best WLB?",
        "Is Rust in demand for backend?",
        "What tools do data engineers use?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex

    st.divider()
    cols = st.columns(3)
    cols[0].metric("LinkedIn", "360K")
    cols[1].metric("GD", "200K")
    cols[2].metric("SO", "64K")

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <h1 style="color:#e8eaf0;font-weight:600;font-size:2rem;margin:0;">
        Career Intelligence
    </h1>
    <p style="color:#8b92a5;font-size:1rem;margin:0.25rem 0 0 0;">
        Grounded answers from LinkedIn · Glassdoor · Stack Overflow
    </p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about jobs, salaries, skills, or company culture...")
if "query" in st.session_state:
    query = st.session_state.pop("query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    embedder, collections, bm25_data, reranker = load_all()

    active = {}
    if use_linkedin:      active["linkedin_jobs"]        = collections["linkedin_jobs"]
    if use_glassdoor:     active["glassdoor_reviews"]    = collections["glassdoor_reviews"]
    if use_stackoverflow: active["stackoverflow_survey"] = collections["stackoverflow_survey"]

    if not active:
        st.warning("Select at least one source in the sidebar.")
        st.stop()

    with st.spinner("Searching 625k entries..."):
        chunks, stats = retrieve(
            query, active, embedder, bm25_data, reranker,
            use_rerank=use_rerank, top_k=50, final_k=top_k
        )

    st.caption(
        f"🔵 Dense: {stats['dense']} · "
        f"🟡 Sparse: {stats['sparse']} · "
        f"🔀 After RRF: {stats['fused']} · "
        f"✅ Final: {stats['final']}"
    )

    with st.chat_message("assistant"):
        prompt   = build_prompt(query, chunks)
        response = st.write_stream(stream_ollama(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.expander(f"📚 {len(chunks)} sources retrieved", expanded=True):
        source_display = {
            "linkedin_jobs"       : "💼 LinkedIn",
            "glassdoor_reviews"   : "⭐ Glassdoor",
            "stackoverflow_survey": "📊 Stack Overflow",
        }
        unique_sources = list({c["source"] for c in chunks})
        tab_labels     = ["All"] + [source_display.get(s, s) for s in unique_sources]
        tabs           = st.tabs(tab_labels)
        all_chunks = chunks

        def render_chunks(chunk_list):
            for i, c in enumerate(chunk_list):
                src  = c["source"]
                meta = c["meta"]
                url  = meta.get("source_url", "#")

                if src == "linkedin_jobs":
                    badge = "🔵 LinkedIn"
                    title = f"{meta.get('title','?')} @ {meta.get('company','?')}"
                    sub   = meta.get('location','?')
                    if meta.get("salary") not in [None, "N/A", "nan"]:
                        try: sub += f" · 💰 ${float(meta['salary']):,.0f}"
                        except: pass
                elif src == "glassdoor_reviews":
                    badge = "🟢 Glassdoor"
                    title = f"{meta.get('job_title','?')} @ {meta.get('firm','?')}"
                    sub   = f"★ {meta.get('rating','?')}/5 · {meta.get('date','')}"
                else:
                    badge = "🟠 Stack Overflow"
                    title = meta.get("role", "Developer")
                    country  = meta.get('country',  '') or '?'
                    yrs      = meta.get('years_exp','') or '?'
                    sub      = f"{country} · {yrs} yrs exp"

                score_key = "rerank_score" if "rerank_score" in c else "rrf_score"
                score = round(c.get(score_key, 0), 3)

                # update the caption to clarify what the score means
                if "rerank_score" in c:
                    score_label = f"rerank: `{score}`"
                else:
                    score_label = f"rrf: `{score}`"

                with st.container():
                    c1, c2 = st.columns([6, 1])
                    with c1:
                        st.markdown(f"**{badge}** — {title}")
                        st.caption(f"{sub} · {score_label}")
                    with c2:
                        st.link_button("↗", url)
                    with st.expander("View chunk"):
                        st.code(c["text"], language=None)
                    st.divider()

        with tabs[0]:
            render_chunks(all_chunks)

        for i, src_name in enumerate(list({c["source"] for c in chunks})):
            with tabs[i + 1]:
                render_chunks([c for c in chunks if c["source"] == src_name])