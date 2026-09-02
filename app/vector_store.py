import hashlib
import json
import os


def get_file_hash(file_path):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(4096)

            if not data:
                break

            hasher.update(data)

    return hasher.hexdigest()


def save_embeddings(
    chunks,
    embeddings,
    document_hash,
    file_path="data/embeddings.json"
):
    data = {
        "document_hash": document_hash,
        "items": []
    }

    for chunk, embedding in zip(chunks, embeddings):
        data["items"].append(
            {
                "chunk": chunk,
                "embedding": embedding
            }
        )

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file)

    print("Embeddings saved successfully.")


def load_embeddings(
    document_hash,
    file_path="data/embeddings.json"
):
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("document_hash") != document_hash:
        print("PDF changed. Creating new embeddings...")
        return None

    print("Saved embeddings loaded successfully.")

    return data["items"]