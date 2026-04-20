import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHROMA_PATH = "./chroma_db"
COLLECTION  = "stackoverflow_survey"
BATCH_SIZE  = 256

print("Loading StackOverflow CSV...")
df = pd.read_csv("data/stackoverflow/survey_results_public.csv", on_bad_lines="skip")
print(f"Raw rows: {len(df)}")

def row_to_text(row):
    parts = []
    for label, col in [
        ("Developer type",      "DevType"),
        ("Years experience",    "YearsCodePro"),
        ("Country",             "Country"),
        ("Education",           "EdLevel"),
        ("Remote work",         "RemoteWork"),
        ("Industry",            "Industry"),
        ("Employment",          "Employment"),
        ("Org size",            "OrgSize"),
    ]:
        val = row.get(col, "")
        if val and str(val) != "nan":
            parts.append(f"{label}: {val}.")

    for label, col in [
        ("Languages used",      "LanguageHaveWorkedWith"),
        ("Languages admired",   "LanguageAdmired"),
        ("Wants to learn",      "LanguageWantToWorkWith"),
        ("Databases used",      "DatabaseHaveWorkedWith"),
        ("Frameworks used",     "WebframeHaveWorkedWith"),
        ("Platforms used",      "PlatformHaveWorkedWith"),
        ("AI tools used",       "AISearchDevHaveWorkedWith"),
        ("Collab tools",        "NEWCollabToolsHaveWorkedWith"),
        ("Misc tech",           "MiscTechHaveWorkedWith"),
    ]:
        val = row.get(col, "")
        if val and str(val) != "nan":
            parts.append(f"{label}: {val}.")

    salary = row.get("ConvertedCompYearly", "")
    if salary and str(salary) not in ["nan", "0"]:
        try:
            parts.append(f"Annual salary: ~${int(float(salary)):,} USD.")
        except:
            pass

    for label, col in [
        ("Job satisfaction",    "JobSat"),
        ("AI sentiment",        "AISent"),
        ("AI benefit view",     "AIBen"),
        ("AI accuracy view",    "AIAcc"),
    ]:
        val = row.get(col, "")
        if val and str(val) != "nan":
            parts.append(f"{label}: {val}.")

    return " ".join(parts)

print("Building chunks...")
all_chunks, all_ids, all_metadata = [], [], []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    text = row_to_text(row)
    if len(text) < 60:
        continue
    all_chunks.append(text)
    all_ids.append(f"so_{idx}")
    all_metadata.append({
        "source"    : "stackoverflow",
        "role"      : str(row.get("DevType", "N/A"))[:100],
        "country"   : str(row.get("Country", "N/A")),
        "years_exp" : str(row.get("YearsCodePro", "N/A")),
        "salary"    : str(row.get("ConvertedCompYearly", "N/A")),
        "source_url": "https://survey.stackoverflow.co/2024/",
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
    end  = min(start + BATCH_SIZE, len(all_chunks))
    vecs = model.encode(
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

print(f"\nDone! {collection.count()} StackOverflow chunks stored.")
