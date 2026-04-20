

**Multi-source Retrieval-Augmented Generation for Career Intelligence**

> CS 6120 Natural Language Processing — Final Project  
> Northeastern University, Khoury College of Computer Science  
> Shang Chen | Spring 2026

CareerRAG answers natural language career questions by grounding responses in 625,000 real data points from LinkedIn job postings, Glassdoor employee reviews, and the Stack Overflow Developer Survey. All inference is fully local — no external API calls.

**Live demo:** http://34.173.195.254:8501

---

## System Overview

```
Query
  ├── Dense retrieval   (ChromaDB + MiniLM embeddings)
  ├── Sparse retrieval  (BM25 keyword matching)
  └── RRF fusion → Cross-encoder reranking → Qwen 3.5 → Answer + Citations
```

**Data sources:**

| Source | Dataset | Chunks |
|---|---|---|
| LinkedIn Job Postings 2023-24 | arshkon/linkedin-job-postings | 360,462 |
| Glassdoor Employee Reviews | davidgauthier/glassdoor-job-reviews | 200,000 |
| Stack Overflow Survey 2024 | berkayalan/stack-overflow-annual-developer-survey-2024 | 64,560 |
| **Total** | | **625,022** |

---

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA 12.4+ (tested on L4 24GB)
- 50GB+ disk space
- Kaggle API credentials

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/shang/careerrag
cd careerrag
```

### 2. Add Kaggle credentials

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download datasets

```bash
kaggle datasets download -d arshkon/linkedin-job-postings
kaggle datasets download -d davidgauthier/glassdoor-job-reviews
kaggle datasets download -d berkayalan/stack-overflow-annual-developer-survey-2024

unzip linkedin-job-postings.zip -d data/
unzip glassdoor-job-reviews.zip -d data/glassdoor/
unzip stack-overflow-annual-developer-survey-2024.zip -d data/stackoverflow/
```

### 4. Install Python dependencies (on host)

```bash
pip install chromadb sentence-transformers rank-bm25 \
    langchain-text-splitters pandas tqdm ragas \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
    --break-system-packages
```

### 5. Run ingestion (builds ChromaDB + BM25 index)

```bash
# Embed and index all three sources (~45 min total on L4)
python3 ingest.py               # LinkedIn  → 360k chunks
python3 ingest_glassdoor.py     # Glassdoor → 200k chunks
python3 ingest_stackoverflow.py # SO Survey → 64k chunks

# Build BM25 sparse index (~5 min)
python3 build_bm25.py
```

### 6. Build and run Docker container

```bash
./docker-startup build
./docker-startup deploy-gpu
```

Navigate to `http://localhost:8501`

---

## File Structure

```
careerrag/
├── Dockerfile                  # Container definition
├── docker-startup              # Build/deploy helper script
├── entrypoint.sh               # Container startup (Ollama + Streamlit)
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml             # Streamlit dark theme config
│
├── app.py                      # Main Streamlit RAG application
├── ingest.py                   # LinkedIn ingestion pipeline
├── ingest_glassdoor.py         # Glassdoor ingestion pipeline
├── ingest_stackoverflow.py     # Stack Overflow ingestion pipeline
├── build_bm25.py               # BM25 sparse index builder
├── evaluate_ragas.py           # RAGAS automated evaluation
│
├── data/                       # Raw CSV datasets (not in repo)
│   ├── postings.csv
│   ├── glassdoor/
│   │   └── glassdoor_reviews.csv
│   └── stackoverflow/
│       └── survey_results_public.csv
│
├── chroma_db/                  # ChromaDB vector store (not in repo)
│   ├── linkedin_jobs/
│   ├── glassdoor_reviews/
│   └── stackoverflow_survey/
│
└── bm25_index.pkl              # BM25 index (not in repo, built by build_bm25.py)
```

---

## RAG Pipeline Details

### Ingestion

Each source goes through: **load → clean → chunk → enrich → embed → store**

LinkedIn postings are enriched by joining 5 supplementary CSV files:
- `jobs/job_skills.csv` + `mappings/skills.csv` → decoded skill names
- `jobs/job_industries.csv` + `mappings/industries.csv` → industry labels
- `jobs/benefits.csv` → benefit packages
- `companies/companies.csv` → company descriptions and size

Stack Overflow structured responses are converted to natural language sentences before embedding.

### Retrieval

```
Dense:  query → MiniLM embed → ChromaDB cosine search → top-50
Sparse: query → BM25 tokenize → BM25Okapi score → top-50
Fusion: RRF(dense, sparse, k=60) → top-20
Rerank: cross-encoder/ms-marco-MiniLM-L-6-v2 → top-5
```

### Models

| Role | Model | Size |
|---|---|---|
| Embedding | all-MiniLM-L6-v2 | 80MB |
| Reranker | ms-marco-MiniLM-L-6-v2 | 80MB |
| LLM | Qwen 3.5 (via Ollama) | 6.6GB |

---

## Evaluation

Run RAGAS evaluation (requires Ollama running):

```bash
python3 evaluate_ragas.py
# outputs ragas_results.csv with faithfulness and answer_relevancy scores
```

---

## Docker Notes

The `docker-startup` script mounts the current directory as `/root` inside the container:

```bash
docker run --rm --name run-llama3 -d --gpus=all \
  -v $PWD:/root \
  -p 11434:11434 -p 8501:8501 \
  run-llama3
```

This means `chroma_db/`, `bm25_index.pkl`, and `app.py` are shared between host and container in real time — no rebuild needed when editing `app.py`.

---

## Dataset Citation

```
LinkedIn Job Postings (2023-2024): Arsh Koneru
https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

Glassdoor Job Reviews: David Gauthier
https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews

Stack Overflow Annual Developer Survey 2024: Berkay Alan
https://www.kaggle.com/datasets/berkayalan/stack-overflow-annual-developer-survey-2024
```

---

## Tech Stack

`Python 3.12` · `Streamlit` · `ChromaDB` · `sentence-transformers` · `rank-bm25` · `Ollama` · `Qwen 3.5` · `Docker` · `NVIDIA L4` · `GCP`
