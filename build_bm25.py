import chromadb
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re

CHROMA_PATH = "./chroma_db"
BM25_PATH   = "./bm25_index.pkl"
BATCH       = 5000

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return text.split()

client = chromadb.PersistentClient(path=CHROMA_PATH)

all_docs  = []
all_ids   = []
all_metas = []

for col_name in ["linkedin_jobs", "glassdoor_reviews", "stackoverflow_survey"]:
    print(f"Loading {col_name}...")
    col   = client.get_collection(col_name)
    total = col.count()

    for start in tqdm(range(0, total, BATCH)):
        result = col.get(
            limit=BATCH,
            offset=start,
            include=["documents", "metadatas"]
        )
        all_docs.extend(result["documents"])
        all_ids.extend(result["ids"])
        all_metas.extend(result["metadatas"])

print(f"Total docs loaded: {len(all_docs)}")
print("Tokenizing...")
tokenized = [tokenize(d) for d in tqdm(all_docs)]

print("Building BM25 index...")
bm25 = BM25Okapi(tokenized)

print("Saving...")
with open(BM25_PATH, "wb") as f:
    pickle.dump({
        "bm25"  : bm25,
        "ids"   : all_ids,
        "docs"  : all_docs,
        "metas" : all_metas,
    }, f)

print(f"Done! BM25 index saved to {BM25_PATH}")
print(f"Index covers {len(all_docs):,} chunks")