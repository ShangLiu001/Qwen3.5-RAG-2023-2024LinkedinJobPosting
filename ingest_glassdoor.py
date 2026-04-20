import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re

CHROMA_PATH = "./chroma_db"
COLLECTION  = "glassdoor_reviews"
BATCH_SIZE  = 256
SAMPLE_SIZE = 200_000

print("Loading Glassdoor CSV...")
df = pd.read_csv("data/glassdoor/glassdoor_reviews.csv", on_bad_lines="skip")
print(f"Raw rows: {len(df)}")

df = df.dropna(subset=["pros", "cons"], how="all")
df["pros"]     = df["pros"].fillna("").astype(str).str.strip()
df["cons"]     = df["cons"].fillna("").astype(str).str.strip()
df["headline"] = df["headline"].fillna("").astype(str).str.strip()

df = df[~((df["pros"].str.lower().isin(["na","n/a","none","nothing",""])) &
          (df["cons"].str.lower().isin(["na","n/a","none","nothing",""])))]

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

print(f"Rows after cleaning: {len(df)}")

def clean(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_chunk(row):
    parts = []
    firm     = clean(row.get("firm", "Unknown"))
    title    = clean(row.get("job_title", "Employee"))
    headline = clean(row.get("headline", ""))
    pros     = clean(row.get("pros", ""))
    cons     = clean(row.get("cons", ""))

    parts.append(f"Company: {firm}")
    parts.append(f"Role: {title}")

    for label, col in [
        ("Overall rating",    "overall_rating"),
        ("Work/life balance", "work_life_balance"),
        ("Culture & values",  "culture_values"),
        ("Comp & benefits",   "comp_benefits"),
        ("Career growth",     "career_opp"),
        ("Senior mgmt",       "senior_mgmt"),
    ]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip() not in ["", "nan"]:
            parts.append(f"{label}: {val}/5")

    if headline: parts.append(f"Summary: {headline}")
    if pros:     parts.append(f"Pros: {pros}")
    if cons:     parts.append(f"Cons: {cons}")
    return "\n".join(parts)

print("Building chunks...")
all_chunks, all_ids, all_metadata = [], [], []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    chunk = build_chunk(row)
    if len(chunk) < 80:
        continue
    firm = str(row.get("firm", "company")).lower().replace(" ", "-")
    source_url = f"https://www.glassdoor.com/Reviews/{firm}-reviews.htm"

    all_chunks.append(chunk)
    all_ids.append(f"gd_{idx}")
    all_metadata.append({
        "source"           : "glassdoor",
        "firm"             : str(row.get("firm", "N/A")),
        "job_title"        : str(row.get("job_title", "N/A")),
        "rating"           : str(row.get("overall_rating", "N/A")),
        "work_life_balance": str(row.get("work_life_balance", "N/A")),
        "culture"          : str(row.get("culture_values", "N/A")),
        "comp_benefits"    : str(row.get("comp_benefits", "N/A")),
        "date"             : str(row.get("date_review", "N/A")),
        "source_url"       : source_url,
    })

print(f"Total chunks: {len(all_chunks)}")

print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    client.delete_collection(COLLECTION)
    print("Deleted old collection.")
except:
    pass

collection = client.create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

print("Embedding and inserting...")
for start in tqdm(range(0, len(all_chunks), BATCH_SIZE)):
    end   = min(start + BATCH_SIZE, len(all_chunks))
    vecs  = model.encode(
        all_chunks[start:end],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    collection.add(
        ids=all_ids[start:end],
        documents=all_chunks[start:end],
        embeddings=vecs,
        metadatas=all_metadata[start:end],
    )

print(f"\nDone! {collection.count()} Glassdoor chunks stored.")
