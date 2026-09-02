import math


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


def retrieve_top_chunks(
    question_embedding,
    chunk_embeddings,
    chunks,
    top_k=3
):
    similarities = []

    for index, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        similarities.append((score, index))

    similarities.sort(reverse=True)

    top_results = similarities[:top_k]

    relevant_chunks = []

    for score, index in top_results:
        relevant_chunks.append(
            {
                "score": score,
                "chunk": chunks[index]
            }
        )

    return relevant_chunks