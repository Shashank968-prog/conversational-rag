from rag_pipeline import (
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
# 1. Load Documents
# =========================================================

documents = load_documents()

print("\nNumber of pages:", len(documents))


# =========================================================
# 2. Split Documents
# =========================================================

chunks = split_documents(documents)

print("Number of chunks:", len(chunks))


# =========================================================
# 3. Create / Load Vector Store
# =========================================================

vector_store = get_vector_store(chunks)

print("\nVector store created/loaded successfully!")


# =========================================================
# 4. Create Retriever
# =========================================================

retriever = create_retriever(vector_store)

print("Retriever created successfully!")


# =========================================================
# 5. Conversation History
# =========================================================

history = []


# =========================================================
# 6. Interactive Conversational RAG
# =========================================================

while True:

    question = input("\nYou: ")


    # -----------------------------------------------------
    # Exit
    # -----------------------------------------------------

    if question.lower() == "exit":

        print("Goodbye!")

        break


    # =====================================================
    # 7. Show Conversation History
    # =====================================================

    if history:

        print("\n===== Conversation History =====")

        print(format_history(history))


    # =====================================================
    # 8. Rewrite Question
    # =====================================================

    standalone_question = rewrite_question(
        question,
        history
    )

    print("\n===== Rewritten Question =====")

    print(standalone_question)


    # =====================================================
    # 9. Retrieve Relevant Chunks
    # =====================================================

    results = retriever.invoke(
        standalone_question
    )

    print(
        "\nNumber of retrieved chunks:",
        len(results)
    )


    # =====================================================
    # 10. Display Retrieved Chunks
    # =====================================================

    for i, doc in enumerate(results):

        print(
            f"\n===== Retrieved Chunk {i + 1} ====="
        )

        print(doc.page_content)


    # =====================================================
    # 11. Format Retrieved Documents
    # =====================================================

    context = format_docs(results)

    print("\n===== Context =====")

    print(context)


    # =====================================================
    # 12. Generate Final Answer
    # =====================================================

    answer = generate_answer(
        question,
        context
    )


    # =====================================================
    # 13. Display Final Answer
    # =====================================================

    print("\n===== Final Answer =====")

    print(answer)


    # =====================================================
    # 14. Save Conversation
    # =====================================================

    history.append({
        "question": question,
        "answer": answer
    })