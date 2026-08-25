from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os

from langchain_community.document_loaders import PyPDFLoader

from app.rag.rag_pipeline import (
    load_documents,
    split_documents,
    get_vector_store,
    create_retriever,
    format_docs,
    generate_answer,
    format_history,
    rewrite_question
)


# =========================================================
# Global RAG Variables
# =========================================================

retriever = None
vector_store = None

# Conversation history, keyed by session_id so different
# users/tabs don't share the same conversation.
history_by_session: dict[str, list] = {}


# =========================================================
# Project Paths
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)

DOCUMENTS_DIR = os.path.join(
    BASE_DIR,
    "documents"
)


# =========================================================
# Create Documents Directory
# =========================================================

os.makedirs(
    DOCUMENTS_DIR,
    exist_ok=True
)


# =========================================================
# FastAPI Lifespan
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global retriever
    global vector_store

    print("========================================")
    print("Starting Conversational RAG")
    print("========================================")

    # -----------------------------------------------------
    # 1. Load Documents
    # -----------------------------------------------------

    print("Loading documents...")

    documents = load_documents()

    print(
        "Number of documents/pages:",
        len(documents)
    )

    # -----------------------------------------------------
    # 2. Split Documents
    # -----------------------------------------------------

    print("Splitting documents...")

    chunks = split_documents(
        documents
    )

    print(
        "Number of chunks:",
        len(chunks)
    )

    # -----------------------------------------------------
    # 3. Vector Store
    # -----------------------------------------------------

    print("Loading / creating vector store...")

    vector_store = get_vector_store(
        chunks
    )

    print("Vector store ready!")

    # -----------------------------------------------------
    # 4. Retriever
    # -----------------------------------------------------

    print("Creating retriever...")

    retriever = create_retriever(
        vector_store
    )

    print("Retriever ready!")

    print("========================================")
    print("RAG pipeline ready!")
    print("========================================")

    yield

    # -----------------------------------------------------
    # Shutdown
    # -----------------------------------------------------

    print("Shutting down RAG application...")


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="Conversational RAG API",
    description="Conversational RAG using FastAPI, Chroma and Gemini",
    version="1.0.0",
    lifespan=lifespan
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# Frontend
# =========================================================

app.mount(
    "/frontend",
    StaticFiles(
        directory=FRONTEND_DIR
    ),
    name="frontend"
)


# =========================================================
# Request Model
# =========================================================

class ChatRequest(BaseModel):

    question: str
    session_id: str


# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Conversational RAG API is running"
    }


# =========================================================
# Frontend Endpoint
# =========================================================

@app.get("/frontend")
def frontend():

    return FileResponse(
        os.path.join(
            FRONTEND_DIR,
            "index.html"
        )
    )


# =========================================================
# PDF Upload Endpoint
# =========================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    global vector_store
    global retriever

    # =====================================================
    # STEP 1: Check File Type
    # =====================================================

    if file.content_type != "application/pdf":

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # =====================================================
    # STEP 2: Check Filename
    # =====================================================

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    # =====================================================
    # STEP 3: Create File Path
    # =====================================================

    file_path = os.path.join(
        DOCUMENTS_DIR,
        file.filename
    )

    # =====================================================
    # STEP 4: Save Uploaded PDF
    # =====================================================

    with open(
        file_path,
        "wb"
    ) as buffer:

        buffer.write(
            await file.read()
        )

    print(
        f"PDF uploaded: {file.filename}"
    )

    # =====================================================
    # STEP 5: Load Uploaded PDF
    # =====================================================

    loader = PyPDFLoader(
        file_path
    )

    documents = loader.load()

    # =====================================================
    # STEP 6: Add Metadata
    # =====================================================

    for document in documents:

        document.metadata["filename"] = (
            file.filename
        )

    # =====================================================
    # STEP 7: Split PDF
    # =====================================================

    chunks = split_documents(
        documents
    )

    print(
        "Number of new chunks:",
        len(chunks)
    )

    # =====================================================
    # STEP 8: Add Chunks to Existing Chroma
    # =====================================================

    vector_store.add_documents(
        chunks
    )

    print(
        "New chunks added to ChromaDB."
    )

    # =====================================================
    # STEP 9: Recreate Retriever
    # =====================================================

    retriever = create_retriever(
        vector_store
    )

    print(
        "Retriever updated."
    )

    # =====================================================
    # STEP 10: Return Response
    # =====================================================

    return {
        "message": "PDF uploaded and indexed successfully.",
        "filename": file.filename,
        "pages": len(documents),
        "chunks": len(chunks)
    }


# =========================================================
# Chat Endpoint
# =========================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    # =====================================================
    # Check Retriever
    # =====================================================

    if retriever is None:

        return {
            "error": "RAG pipeline is not ready yet."
        }

    # =====================================================
    # STEP 1: Get This Session's History
    # =====================================================

    session_history = history_by_session.setdefault(
        request.session_id,
        []
    )

    # =====================================================
    # STEP 2: Get User Question
    # =====================================================

    question = request.question

    # =====================================================
    # STEP 3: Rewrite Question
    # =====================================================

    standalone_question = rewrite_question(
        question,
        session_history
    )

    print("\nOriginal Question:")
    print(question)

    print("\nRewritten Question:")
    print(standalone_question)

    # =====================================================
    # STEP 4: Retrieve Relevant Chunks
    # =====================================================

    results = retriever.invoke(
        standalone_question
    )

    print(
        "\nNumber of retrieved chunks:",
        len(results)
    )

    # =====================================================
    # STEP 5: Display Retrieved Chunks
    # =====================================================

    for i, doc in enumerate(results):

        print(
            f"\n===== Retrieved Chunk {i + 1} ====="
        )

        print(
            doc.page_content
        )

    # =====================================================
    # STEP 6: Format Context
    # =====================================================

    context = format_docs(
        results
    )

    # =====================================================
    # STEP 7: Generate Answer
    # =====================================================

    answer = generate_answer(
        question,
        context
    )

    # =====================================================
    # STEP 8: Save Conversation
    # =====================================================

    session_history.append({
        "question": question,
        "answer": answer
    })

    # =====================================================
    # STEP 9: Return Response
    # =====================================================

    return {
        "question": question,
        "rewritten_question": standalone_question,
        "answer": answer
    }