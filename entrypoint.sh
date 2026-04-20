#!/bin/bash
set -e

echo "=== CareerRAG Startup ==="

# Start Ollama
echo "[1/5] Starting Ollama server..."
ollama serve &
sleep 8
echo "[2/5] Pulling Qwen 3.5..."
ollama pull qwen3.5

# Download pre-built indices if missing
if [ ! -d "chroma_db" ] || [ ! -f "bm25_index.pkl" ]; then
    echo "[3/5] Downloading pre-built indices from Google Drive (~2.5GB)..."
    pip install gdown --break-system-packages -q
    gdown --id 1RMIzJ_yQE-5OFGekAEvjuN2LjnlfMV__ -O careerrag_indices.tar.gz
    echo "Extracting..."
    tar -xzf careerrag_indices.tar.gz
    rm -f careerrag_indices.tar.gz
    echo "Indices ready."
else
    echo "[3/5] Indices already present, skipping download."
fi

echo "[4/5] Verifying setup..."
python3 -c "
import chromadb, pickle
c = chromadb.PersistentClient('./chroma_db')
total = sum(c.get_collection(n).count() for n in ['linkedin_jobs','glassdoor_reviews','stackoverflow_survey'])
print(f'ChromaDB: {total:,} chunks')
with open('bm25_index.pkl','rb') as f:
    import pickle; d = pickle.load(f)
print(f'BM25: {len(d[\"docs\"]):,} docs')
"

echo "[5/5] Starting Streamlit..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
