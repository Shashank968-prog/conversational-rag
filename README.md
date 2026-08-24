# Conversational RAG

A conversational Retrieval-Augmented Generation (RAG) application built using FastAPI, Google Gemini, ChromaDB, and a web frontend.

The application allows users to ask questions about information contained in a PDF document and supports conversational follow-up questions using conversation history and question rewriting.

## Features

- PDF document loading
- Document chunking
- Gemini embeddings
- ChromaDB vector store
- Similarity-based retrieval
- Top-k document retrieval
- Question rewriting for conversational queries
- Conversation history
- Gemini-powered answer generation
- FastAPI REST API
- Interactive Swagger API documentation
- HTML, CSS, and JavaScript frontend
- CORS support

## Architecture

```text
User
 |
 v
Frontend
 |
 | POST /chat
 v
FastAPI
 |
 v
Question Rewriting
 |
 v
Retriever
 |
 v
ChromaDB
 |
 v
Relevant Document Chunks
 |
 v
Context
 |
 v
Google Gemini
 |
 v
Generated Answer
 |
 v
Frontend

**Technologies Used**
Python
FastAPI
LangChain
Google Gemini
Gemini Embeddings
ChromaDB
PyPDFLoader
HTML
CSS
JavaScript
Uvicorn

**Project Structure**

conversational-rag/
│
├── app/
│   ├── main.py
│   │
│   ├── rag/
│   │   ├── rag_pipeline.py
│   │   └── test_pipeline.py
│   │
│   └── services/
│
├── documents/
│   └── python_rag_notes.pdf
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── requirements.txt
├── .gitignore
└── README.md
