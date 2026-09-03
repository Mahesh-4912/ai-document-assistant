import hashlib
import os
import tempfile

import streamlit as st

from app.agent import choose_action
from app.document_loader import (
    read_pdf,
    split_pages,
)
from app.embeddings import get_embedding
from app.chroma_store import (
    document_exists,
    save_to_chroma,
    search_chroma,
    get_document_chunks,
)
from app.llm import (
    generate_answer,
    summarize_document,
)


def get_file_hash(file_path):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(4096)

            if not data:
                break

            hasher.update(data)

    return hasher.hexdigest()


st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")

st.write(
    "Ask questions or summarize PDF documents "
    "using RAG, Agentic AI, Ollama and ChromaDB."
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name

    document_hash = get_file_hash(
        temp_path
    )

    if not document_exists(
        document_hash
    ):

        with st.spinner(
            "Processing document..."
        ):

            pages = read_pdf(
                temp_path
            )

            chunks = split_pages(
                pages
            )

            embeddings = []

            for chunk in chunks:
                embeddings.append(
                    get_embedding(
                        chunk["text"]
                    )
                )

            save_to_chroma(
                chunks=chunks,
                embeddings=embeddings,
                document_hash=document_hash,
                file_name=uploaded_file.name
            )

        st.success(
            "Document processed successfully."
        )

    else:
        st.success(
            "Document already available in database."
        )

    question = st.text_input(
        "Ask something about the document",
        placeholder=(
            "Example: What is my experience? "
            "or Summarize this document"
        )
    )

    if st.button("Run"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "AI Agent is working..."
            ):

                action = choose_action(
                    question
                )

                st.caption(
                    f"Agent selected: {action}"
                )

                # -------------------------
                # SUMMARIZE ACTION
                # -------------------------

                if action == "SUMMARIZE":

                    document_chunks = (
                        get_document_chunks(
                            document_hash
                        )
                    )

                    # Limit context because
                    # local model is small
                    selected_chunks = (
                        document_chunks[:12]
                    )

                    context = "\n\n".join(
                        item["chunk"]
                        for item
                        in selected_chunks
                    )

                    answer = summarize_document(
                        context
                    )

                    st.subheader(
                        "Document Summary"
                    )

                    st.write(
                        answer
                    )

                # -------------------------
                # ASK ACTION
                # -------------------------

                else:

                    question_embedding = (
                        get_embedding(
                            question
                        )
                    )

                    results = search_chroma(
                        question_embedding=
                        question_embedding,
                        document_hash=
                        document_hash,
                        top_k=3
                    )

                    if not results:

                        st.warning(
                            "No relevant "
                            "information found."
                        )

                    else:

                        context = "\n\n".join(
                            result["chunk"]
                            for result
                            in results
                        )

                        answer = generate_answer(
                            question=question,
                            context=context
                        )

                        st.subheader(
                            "Answer"
                        )

                        st.write(
                            answer
                        )

                        st.subheader(
                            "Sources"
                        )

                        for index, result in enumerate(
                            results,
                            start=1
                        ):

                            with st.expander(
                                f"Source {index} "
                                f"- Page "
                                f"{result['page_number']}"
                            ):

                                st.write(
                                    result[
                                        "chunk"
                                    ]
                                )

                                st.caption(
                                    "Distance: "
                                    f"{round(result['distance'], 4)}"
                                )

    try:
        os.unlink(
            temp_path
        )

    except PermissionError:
        pass