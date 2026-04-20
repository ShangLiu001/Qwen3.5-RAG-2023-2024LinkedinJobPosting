import pandas as pd
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION  = "linkedin_jobs"
BATCH_SIZE  = 256
CHUNK_SIZE  = 1000   # was 500 — bigger chunks = fewer splits
MAX_CHUNKS_PER_POST = 3   # add this new limit
MAX_ROWS    = None

# ── Load main table ───────────────────────────────────────────────────────────
print("Loading postings...")
df = pd.read_csv("data/postings.csv")
df = df.dropna(subset=["description"])
df = df[df["description"].str.strip().str.len() > 100]
if MAX_ROWS:
    df = df.head(MAX_ROWS)
print(f"Postings after cleaning: {len(df)}")

# ── Load and build lookup tables ──────────────────────────────────────────────
print("Loading lookup tables...")

# skills: job_id → comma-separated skill names
skills_map   = pd.read_csv("data/mappings/skills.csv")
job_skills   = pd.read_csv("data/jobs/job_skills.csv")
job_skills   = job_skills.merge(skills_map, on="skill_abr", how="left")
skills_lookup = (
    job_skills.groupby("job_id")["skill_name"]
    .apply(lambda x: ", ".join(x.dropna()))
    .to_dict()
)

# industries: job_id → industry name
industry_map   = pd.read_csv("data/mappings/industries.csv")
job_industries = pd.read_csv("data/jobs/job_industries.csv")
job_industries = job_industries.merge(industry_map, on="industry_id", how="left")
industry_lookup = (
    job_industries.groupby("job_id")["industry_name"]
    .apply(lambda x: ", ".join(x.dropna()))
    .to_dict()
)

# benefits: job_id → comma-separated benefits
job_benefits = pd.read_csv("data/jobs/benefits.csv")
benefits_lookup = (
    job_benefits.groupby("job_id")["type"]
    .apply(lambda x: ", ".join(x.dropna()))
    .to_dict()
)

# companies: company_id → description, size, url
companies = pd.read_csv("data/companies/companies.csv")
companies = companies.set_index("company_id")

print("Lookups built.")

# ── Helpers ───────────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

def clean(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

COMPANY_SIZES = {
    1: "1-10", 2: "11-50", 3: "51-200", 4: "201-500",
    5: "501-1000", 6: "1001-5000", 7: "5001-10000", 8: "10000+"
}

def build_text(row):
    parts = []
    job_id     = row["job_id"]
    company_id = row.get("company_id")

    # core fields
    for col, label in [
        ("title",                    "Title"),
        ("company_name",             "Company"),
        ("location",                 "Location"),
        ("formatted_experience_level","Experience level"),
        ("formatted_work_type",      "Work type"),
    ]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(f"{label}: {clean(str(val))}")

    # salary
    sal = row.get("normalized_salary")
    period = row.get("pay_period", "")
    if pd.notna(sal) and float(sal) > 0:
        parts.append(f"Salary: ${int(sal):,} ({period})")

    # enriched fields from joins
    skills = skills_lookup.get(job_id, "")
    if skills:
        parts.append(f"Skills required: {skills}")

    industry = industry_lookup.get(job_id, "")
    if industry:
        parts.append(f"Industry: {industry}")

    benefits = benefits_lookup.get(job_id, "")
    if benefits:
        parts.append(f"Benefits: {benefits}")

    # company description
    if pd.notna(company_id):
        try:
            co = companies.loc[int(company_id)]
            co_desc = clean(str(co.get("description", "")))[:300]
            co_size = COMPANY_SIZES.get(int(co.get("company_size", 0)), "")
            if co_desc:
                parts.append(f"About company: {co_desc}")
            if co_size:
                parts.append(f"Company size: {co_size} employees")
        except:
            pass

    # description last (longest)
    desc = clean(str(row.get("description", "")))
    if desc:
        parts.append(f"Description: {desc}")

    skills_desc = clean(str(row.get("skills_desc", "")))
    if skills_desc and skills_desc != "nan":
        parts.append(f"Skills details: {skills_desc}")

    return "\n".join(parts)

# ── Build chunks ──────────────────────────────────────────────────────────────
print("Building chunks...")
all_chunks, all_ids, all_metadata = [], [], []



for idx, row in tqdm(df.iterrows(), total=len(df)):
    raw    = build_text(row)
    pieces = splitter.split_text(raw) if len(raw) > 1200 else [raw]
    pieces = pieces[:MAX_CHUNKS_PER_POST]   # ← add this line

    for ci, piece in enumerate(pieces):
        source_url = str(row.get("job_posting_url", "https://www.linkedin.com/jobs"))
        all_chunks.append(piece)
        all_ids.append(f"li_{idx}_{ci}")
        all_metadata.append({
            "source"      : "linkedin",
            "title"       : str(row.get("title", "N/A")),
            "company"     : str(row.get("company_name", "N/A")),
            "location"    : str(row.get("location", "N/A")),
            "industry"    : industry_lookup.get(row["job_id"], "N/A"),
            "salary"      : str(row.get("normalized_salary", "N/A")),
            "source_url"  : source_url,
        })

print(f"Total chunks: {len(all_chunks)}")

# ── Embed & store ─────────────────────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"
)

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
    texts = all_chunks[start:end]
    ids   = all_ids[start:end]
    metas = all_metadata[start:end]

    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    collection.add(
        ids=ids, documents=texts,
        embeddings=vecs, metadatas=metas,
    )

print(f"\nDone! {collection.count()} LinkedIn chunks stored.")