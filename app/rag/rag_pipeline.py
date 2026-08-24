from dotenv import load_dotenv

load_dotenv()

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# =========================================================
# Hybrid Search Imports
# =========================================================

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# =========================================================
# CrossEncoder
# =========================================================

from sentence_transformers.cross_encoder import CrossEncoder


# =========================================================
# BASE DIRECTORIES
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DOCUMENTS_DIR = os.path.join(
    BASE_DIR,
    "documents"
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# =========================================================
# Gemini Chat Model
# =========================================================

model = init_chat_model(
    "gemini-3.6-flash",
    model_provider="google_genai"
)


# =========================================================
# 1. Load Documents
# =========================================================

def load_documents():

    documents = []

    for filename in os.listdir(DOCUMENTS_DIR):

        if filename.lower().endswith(".pdf"):

            file_path = os.path.join(
                DOCUMENTS_DIR,
                filename
            )

            loader = PyPDFLoader(file_path)

            pdf_documents = loader.load()

            documents.extend(pdf_documents)

    return documents


# =========================================================
# 2. Split Documents
# =========================================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


# =========================================================
# 3. Gemini Embeddings
# =========================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# =========================================================
# 4. Create Chroma Database
# =========================================================

def create_vector_store(chunks):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vector_store


# =========================================================
# 5. Load Existing Chroma Database
# =========================================================

def load_vector_store():

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vector_store


# =========================================================
# 6. Get Vector Store
# =========================================================

def get_vector_store(chunks):

    if (
        not os.path.exists(CHROMA_PATH)
        or not os.listdir(CHROMA_PATH)
    ):

        print("Chroma database not found.")
        print("Creating Chroma database...")

        vector_store = create_vector_store(
            chunks
        )

    else:

        print("Chroma database found.")
        print("Loading existing Chroma database...")

        vector_store = load_vector_store()

    return vector_store


# =========================================================
# 7. MMR Vector Retriever
# =========================================================

def create_retriever(vector_store):

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 5,
            "lambda_mult": 0.5
        }
    )

    return retriever


# =========================================================
# 8. Hybrid Retriever
# =========================================================

def create_hybrid_retriever(
    vector_store,
    chunks
):

    # -----------------------------------------------------
    # Vector Retriever
    # -----------------------------------------------------

    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 5,
            "lambda_mult": 0.5
        }
    )

    # -----------------------------------------------------
    # BM25 Keyword Retriever
    # -----------------------------------------------------

    bm25_retriever = BM25Retriever.from_documents(
        chunks
    )

    bm25_retriever.k = 3

    # -----------------------------------------------------
    # Combine Vector + BM25
    # -----------------------------------------------------

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever
        ],
        weights=[
            0.7,
            0.3
        ]
    )

    return hybrid_retriever


# =========================================================
# 9. Reranker
# =========================================================

reranker_model = None


def get_reranker():

    global reranker_model

    if reranker_model is None:

        print("Loading CrossEncoder reranker...")

        reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print("CrossEncoder loaded successfully.")

    return reranker_model


# =========================================================
# 10. Rerank Documents
# =========================================================

def rerank_documents(
    question,
    documents,
    top_k=3
):

    if not documents:
        return []

    # -----------------------------------------------------
    # Load reranker only when required
    # -----------------------------------------------------

    reranker = get_reranker()

    # -----------------------------------------------------
    # Create question-document pairs
    # -----------------------------------------------------

    pairs = [
        [
            question,
            doc.page_content
        ]
        for doc in documents
    ]

    # -----------------------------------------------------
    # Calculate relevance scores
    # -----------------------------------------------------

    scores = reranker.predict(
        pairs
    )

    # -----------------------------------------------------
    # Combine documents with scores
    # -----------------------------------------------------

    scored_documents = list(
        zip(
            documents,
            scores
        )
    )

    # -----------------------------------------------------
    # Sort by relevance
    # -----------------------------------------------------

    scored_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # -----------------------------------------------------
    # Return top documents
    # -----------------------------------------------------

    return [
        doc
        for doc, score in scored_documents[:top_k]
    ]


# =========================================================
# 11. Format Documents
# =========================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# =========================================================
# 12. RAG Prompt
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
Answer the question using only the provided context.

If the answer is not available in the context,
say that the information is not available in the documents.

Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================================================
# 13. Generate Answer
# =========================================================

def generate_answer(
    question,
    context
):

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = model.invoke(
        final_prompt
    )

    return response.content[0]["text"]


# =========================================================
# 14. Format Conversation History
# =========================================================

def format_history(history):

    return "\n".join(
        f"User: {item['question']}\n"
        f"AI: {item['answer']}"
        for item in history
    )


# =========================================================
# 15. Question Rewriting Prompt
# =========================================================

rewrite_prompt = ChatPromptTemplate.from_template(
    """
Given the conversation history and the latest user question,
rewrite the latest question as a standalone question.

Use the conversation history to understand references
such as "it", "its", "they", "this", or "that".

Do not answer the question.

Conversation History:
{history}

Latest Question:
{question}

Standalone Question:
"""
)


# =========================================================
# 16. Rewrite Question
# =========================================================

def rewrite_question(
    question,
    history
):

    formatted_history = format_history(
        history
    )

    final_prompt = rewrite_prompt.invoke({
        "history": formatted_history,
        "question": question
    })

    response = model.invoke(
        final_prompt
    )

    return response.content[0]["text"]