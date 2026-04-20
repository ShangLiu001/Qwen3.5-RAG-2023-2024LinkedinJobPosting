import json, gc, torch
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_ollama import ChatOllama, OllamaEmbeddings

print("Loading saved answers...")
with open("ragas_input.json") as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    "question": [d["question"] for d in data],
    "answer"  : [d["answer"]   for d in data],
    "contexts": [d["contexts"] for d in data],
})

print("Configuring local Qwen judge...")
llm = ChatOllama(model="qwen3.5:latest", base_url="http://localhost:11434", temperature=0)
emb = OllamaEmbeddings(model="llama3", base_url="http://localhost:11434")

print("Running RAGAS...")
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm, embeddings=emb)

df = results.to_pandas()
print("\n=== RAGAS RESULTS ===")
print(df[["question","faithfulness","answer_relevancy"]].to_string())
print(f"\nFaithfulness:     {df['faithfulness'].mean():.3f}")
print(f"Answer relevancy: {df['answer_relevancy'].mean():.3f}")
df.to_csv("ragas_results.csv", index=False)
print("Saved ragas_results.csv")
