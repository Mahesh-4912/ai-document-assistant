from app.llm import generate_answer
from app.document_loader import read_pdf, split_text
from app.embeddings import get_embedding
from app.vector_store import (
    get_file_hash,
    load_embeddings,
    save_embeddings,
)
from app.chroma_store import (
    save_to_chroma,
    search_chroma,
)


def main():
    file_path = "data/sample.pdf"
    document_hash = get_file_hash(file_path)

    print("Reading PDF...")

    # 1. Read PDF
    text = read_pdf(file_path)

    if not text:
        print("No text found in the PDF.")
        return

    # 2. Split text into chunks
    chunks = split_text(text)

    print("Total characters:", len(text))
    print("Total chunks:", len(chunks))

    # 3. Load saved embeddings if available
    saved_data = load_embeddings(document_hash)

    if saved_data:
        print("Using saved embeddings...")

        chunks = [
            item["chunk"]
            for item in saved_data
        ]

        chunk_embeddings = [
            item["embedding"]
            for item in saved_data
        ]

    else:
        print("Creating embeddings...")

        chunk_embeddings = []

        for chunk in chunks:
            embedding = get_embedding(chunk)
            chunk_embeddings.append(embedding)

        save_embeddings(
            chunks=chunks,
            embeddings=chunk_embeddings,
            document_hash=document_hash
        )

        print("Embeddings created successfully.")

    # 4. Save vectors to ChromaDB
    save_to_chroma(
        chunks=chunks,
        embeddings=chunk_embeddings,
        document_hash=document_hash
    )

    # 5. Ask user a question
    question = input(
        "\nAsk a question about the PDF: "
    ).strip()

    if not question:
        print("Please enter a valid question.")
        return

    # 6. Convert question into embedding
    question_embedding = get_embedding(question)

    # 7. Search ChromaDB
    results = search_chroma(
        question_embedding=question_embedding,
        top_k=3
    )

    relevant_chunks = []

    print("\n=== ChromaDB Search Results ===")

    for rank, result in enumerate(results, start=1):
        relevant_chunks.append(result["chunk"])

        print(f"\n--- Chunk {rank} ---")
        print(
            f"Distance: "
            f"{round(result['distance'], 4)}"
        )
        print(result["chunk"])

    # 8. Combine retrieved chunks
    context = "\n\n".join(relevant_chunks)

    # 9. Generate final answer
    print("\nGenerating AI answer...")

    answer = generate_answer(
        question=question,
        context=context
    )

    print("\n=== AI Answer ===")
    print(answer)


if __name__ == "__main__":
    main()