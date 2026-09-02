import math
import re

from ollama import chat, embed
from pypdf import PdfReader


# -----------------------------
# 1. Read and clean PDF text
# -----------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            full_text += text + "\n"

    # Remove unnecessary line breaks and extra spaces
    full_text = re.sub(r"\s+", " ", full_text).strip()

    return full_text


# -----------------------------
# 2. Split text into chunks
# -----------------------------
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# 3. Create embedding
# -----------------------------
def get_embedding(text):
    response = embed(
        model="all-minilm",
        input=text
    )

    return response["embeddings"][0]


# -----------------------------
# 4. Calculate similarity
# -----------------------------
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


# -----------------------------
# 5. Generate answer using LLM
# -----------------------------
def generate_answer(question, context):
    prompt = f"""
You are an AI Document Assistant.

Answer the user's question using only the provided document context.

Keep the answer simple, clear, and easy to understand.

If the answer is not available in the context, say:
"I could not find this information in the document."

Document Context:
{context}

User Question:
{question}

Answer:
"""

    response = chat(
        model="qwen3:0.6b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


# -----------------------------
# 6. Main RAG pipeline
# -----------------------------
def main():
    file_path = "data/sample.pdf"

    print("Reading PDF...")

    # Read PDF
    text = read_pdf(file_path)

    if not text:
        print("No text found in the PDF.")
        return

    # Split PDF into chunks
    chunks = split_text(text)

    print("Total characters:", len(text))
    print("Total chunks:", len(chunks))
    print("Creating embeddings...")

    # Create embedding for every chunk
    chunk_embeddings = []

    for chunk in chunks:
        embedding = get_embedding(chunk)
        chunk_embeddings.append(embedding)

    print("Embeddings created successfully.")

    # Ask user a question
    question = input("\nAsk a question about the PDF: ").strip()

    if not question:
        print("Please enter a valid question.")
        return

    # Convert question into embedding
    question_embedding = get_embedding(question)

    # Compare question with every PDF chunk
    similarities = []

    for index, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        similarities.append((score, index))

    # Highest similarity first
    similarities.sort(reverse=True)

    # Get Top 3 relevant chunks
    top_results = similarities[:3]

    relevant_chunks = []

    print("\n=== Top Relevant Chunks ===")

    for rank, (score, index) in enumerate(top_results, start=1):
        chunk = chunks[index]
        relevant_chunks.append(chunk)

        print(f"\n--- Chunk {rank} ---")
        print(f"Similarity Score: {round(score, 4)}")
        print(chunk)

    # Combine Top 3 chunks
    context = "\n\n".join(relevant_chunks)

    print("\nGenerating AI answer...")

    # Send question + relevant context to LLM
    answer = generate_answer(
        question=question,
        context=context
    )

    print("\n=== AI Answer ===")
    print(answer)


if __name__ == "__main__":
    main()