from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os

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
# 1. FastAPI Application
# =========================================================

app = FastAPI(
    title="Conversational RAG API",
    description="Conversational RAG using FastAPI, Chroma and Gemini",
    version="1.0.0"
)


# =========================================================
# 2. CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# 3. Project Paths
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


# =========================================================
# 4. Frontend
# =========================================================

app.mount(
    "/frontend",
    StaticFiles(directory=FRONTEND_DIR),
    name="frontend"
)


# =========================================================
# 5. Request Model
# =========================================================

class ChatRequest(BaseModel):

    question: str


# =========================================================
# 6. Initialize RAG Pipeline
# =========================================================

print("Loading documents...")

documents = load_documents()

print(
    "Number of documents/pages:",
    len(documents)
)


print("Splitting documents...")

chunks = split_documents(documents)

print(
    "Number of chunks:",
    len(chunks)
)


print("Loading / creating vector store...")

vector_store = get_vector_store(chunks)

print("Vector store ready!")


print("Creating retriever...")

retriever = create_retriever(vector_store)

print("Retriever ready!")


# =========================================================
# 7. Conversation History
# =========================================================

history = []


# =========================================================
# 8. Root Endpoint
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Conversational RAG API is running"
    }


# =========================================================
# 9. Frontend Endpoint
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
# 10. Chat Endpoint
# =========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    # =====================================================
    # STEP 1: Get User Question
    # =====================================================

    question = request.question


    # =====================================================
    # STEP 2: Rewrite Question
    # =====================================================

    standalone_question = rewrite_question(
        question,
        history
    )

    print("\nOriginal Question:")
    print(question)

    print("\nRewritten Question:")
    print(standalone_question)


    # =====================================================
    # STEP 3: Retrieve Relevant Chunks
    # =====================================================

    results = retriever.invoke(
        standalone_question
    )

    print(
        "\nNumber of retrieved chunks:",
        len(results)
    )


    # =====================================================
    # STEP 4: Display Retrieved Chunks
    # =====================================================

    for i, doc in enumerate(results):

        print(
            f"\n===== Retrieved Chunk {i + 1} ====="
        )

        print(doc.page_content)


    # =====================================================
    # STEP 5: Format Retrieved Documents
    # =====================================================

    context = format_docs(results)


    # =====================================================
    # STEP 6: Generate Answer
    # =====================================================

    answer = generate_answer(
        question,
        context
    )


    # =====================================================
    # STEP 7: Save Conversation History
    # =====================================================

    history.append({
        "question": question,
        "answer": answer
    })


    # =====================================================
    # STEP 8: Return Response
    # =====================================================

    return {
        "question": question,
        "rewritten_question": standalone_question,
        "answer": answer
    }