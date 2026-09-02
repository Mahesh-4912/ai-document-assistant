import chromadb


client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_or_create_collection(
    name="test_collection"
)

collection.add(
    ids=["1"],
    documents=["Python supports exception handling using try and except."]
)

results = collection.query(
    query_texts=["How do I handle errors in Python?"],
    n_results=1
)

print("ChromaDB is working!")
print(results["documents"])