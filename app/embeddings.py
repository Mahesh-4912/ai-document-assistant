from ollama import embed


def get_embedding(text):
    response = embed(
        model="all-minilm",
        input=text
    )

    return response["embeddings"][0]