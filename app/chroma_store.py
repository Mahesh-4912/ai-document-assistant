import chromadb


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "documents"


def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def save_to_chroma(chunks, embeddings):
    collection = get_collection()

    # Clear old document chunks
    existing_data = collection.get()

    if existing_data["ids"]:
        collection.delete(
            ids=existing_data["ids"]
        )

    ids = [
        f"chunk_{index}"
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    print(
        f"{len(chunks)} chunks saved to ChromaDB successfully."
    )

def search_chroma(question_embedding, top_k=3):
    collection = get_collection()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    relevant_chunks = []

    documents = results["documents"][0]
    distances = results["distances"][0]

    for document, distance in zip(documents, distances):
        relevant_chunks.append(
            {
                "chunk": document,
                "distance": distance
            }
        )

    return relevant_chunks