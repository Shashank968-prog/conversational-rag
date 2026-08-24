# Conversational RAG

A conversational Retrieval-Augmented Generation (RAG) application built using FastAPI, Google Gemini, LangChain, ChromaDB, Hybrid Search, and Reranking.

The application allows users to ask questions about information contained in PDF documents and supports conversational follow-up questions using conversation history and question rewriting.

## Features

- PDF document loading
- Document chunking
- Gemini embeddings
- ChromaDB vector store
- MMR (Maximal Marginal Relevance) retrieval
- Hybrid Search
- Vector-based semantic retrieval
- BM25 keyword-based retrieval
- Ensemble retrieval
- Cross-Encoder reranking
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
Hybrid Search
 |
 +----------------------+
 |                      |
 v                      v
ChromaDB + MMR         BM25
 |                      |
 | Semantic Search      | Keyword Search
 |                      |
 +----------+-----------+
            |
            v
     Ensemble Retrieval
            |
            v
      Candidate Chunks
            |
            v
        Reranking
            |
            v
     Top Relevant Chunks
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
