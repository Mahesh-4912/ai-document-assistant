import chromadb


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "documents"


def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def document_exists(document_hash):
    collection = get_collection()

    result = collection.get(
        where={"document_hash": document_hash}
    )

    return bool(result["ids"])

def save_to_chroma(
    chunks,
    embeddings,
    document_hash,
    file_name
):
    collection = get_collection()

    if document_exists(document_hash):
        return

    ids = [
        f"{document_hash}_{index}"
        for index in range(len(chunks))
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "document_hash": document_hash,
            "file_name": file_name,
            "page_number": chunk["page_number"],
            "chunk_index": index
        }
        for index, chunk in enumerate(chunks)
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def search_chroma(
    question_embedding,
    document_hash,
    top_k=3
):
    collection = get_collection()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where={
            "document_hash": document_hash
        }
    )

    relevant_chunks = []

    if not results["documents"]:
        return relevant_chunks

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for document, distance, metadata in zip(
        documents,
        distances,
        metadatas
    ):
        relevant_chunks.append(
            {
                "chunk": document,
                "distance": distance,
                "file_name": metadata["file_name"],
                "page_number": metadata["page_number"],
                "chunk_index": metadata["chunk_index"]
            }
        )

    return relevant_chunks

def get_document_chunks(document_hash):
    collection = get_collection()

    results = collection.get(
        where={"document_hash": document_hash}
    )

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    chunks = []

    for document, metadata in zip(
        documents,
        metadatas
    ):
        chunks.append(
            {
                "chunk": document,
                "page_number": metadata.get(
                    "page_number",
                    0
                )
            }
        )

    chunks.sort(
        key=lambda item: item["page_number"]
    )

    return chunks