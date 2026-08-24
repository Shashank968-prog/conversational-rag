from dotenv import load_dotenv

load_dotenv()

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


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

    chunks = text_splitter.split_documents(documents)

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

    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):

        print("Chroma database not found.")
        print("Creating Chroma database...")

        vector_store = create_vector_store(chunks)

    else:

        print("Chroma database found.")
        print("Loading existing Chroma database...")

        vector_store = load_vector_store()

    return vector_store

def create_retriever(vector_store):
    retriever=vector_store.as_retriever(
        search_kwargs={"k":3}
    )
    return retriever

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

from langchain_core.prompts import ChatPromptTemplate

# =========================================================
# RAG Prompt
# =========================================================

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.

If the answer is not available in the context,
say that the information is not available in the documents.

Context:
{context}

Question:
{question}

Answer:
""")

# =========================================================
# Generate Answer
# =========================================================

def generate_answer(question, context):

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = model.invoke(final_prompt)

    return response.content[0]["text"]

# =========================================================
# Format Conversation History
# =========================================================

def format_history(history):

    return "\n".join(
        f"User: {item['question']}\n"
        f"AI: {item['answer']}"
        for item in history
    )

# =========================================================
# Question Rewriting Prompt
# =========================================================

rewrite_prompt = ChatPromptTemplate.from_template("""
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
""")
# =========================================================
# Rewrite Question
# =========================================================
def rewrite_question(question,history):
    formatted_history=format_history

    final_prompt=rewrite_prompt.invoke({
        "history":formatted_history,
        "question":question
    })
    response=model.invoke(final_prompt)
    return response.content[0]["text"]