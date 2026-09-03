# AI Document Assistant

AI-powered PDF Document Assistant built using Python, Ollama, ChromaDB, RAG and Agentic AI.

The application allows users to upload PDF documents, ask questions, retrieve relevant document content using semantic vector search and generate contextual answers using a local LLM.

It also includes a simple AI Agent that decides whether the user wants to ask a question or summarize the document.

## Features

- Upload PDF documents
- Extract PDF text
- Split documents into chunks
- Generate embeddings using Ollama
- Store vectors in ChromaDB
- Semantic vector search
- Retrieval-Augmented Generation (RAG)
- AI-generated document answers
- Page-level source references
- Document summarization
- Agentic AI action selection
- Persistent vector database
- Streamlit web interface
- Fully local AI setup

## Tech Stack

- Python
- Streamlit
- Ollama
- Qwen3 0.6B
- all-minilm embeddings
- ChromaDB
- PyPDF
- Git
- GitHub

## Architecture

```text
PDF Upload
    |
    v
Text Extraction
    |
    v
Page-based Chunking
    |
    v
Ollama all-minilm
    |
    v
Embeddings
    |
    v
ChromaDB Vector Database
    |
    v
User Request
    |
    v
AI Agent
   / \
  /   \
ASK   SUMMARIZE
 |
 v
Question Embedding
 |
 v
ChromaDB Semantic Search
 |
 v
Top Relevant Chunks
 |
 v
Qwen3 Local LLM
 |
 v
Final Answer + Sources