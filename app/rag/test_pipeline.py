from rag_pipeline import (
    load_documents,
    split_documents,
    get_vector_store,
    create_hybrid_retriever,
    rerank_documents,
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
# 4. Create Hybrid Retriever
# =========================================================

retriever = create_hybrid_retriever(
    vector_store,
    chunks
)

print("Hybrid Retriever created successfully!")


# =========================================================
# 5. Conversation History
# =========================================================

history = []


# =========================================================
# 6. Interactive Conversational RAG
# =========================================================

while True:

    question = input("\nYou: ")


    # =====================================================
    # Exit
    # =====================================================

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
    # 9. Hybrid Retrieval
    # =====================================================

    candidate_results = retriever.invoke(
        standalone_question
    )

    print(
        "\nNumber of hybrid candidates:",
        len(candidate_results)
    )


    # =====================================================
    # 10. Display Hybrid Results
    # =====================================================

    print("\n===== Hybrid Search Results =====")

    for i, doc in enumerate(candidate_results):

        print(
            f"\n----- Hybrid Result {i + 1} -----"
        )

        print(doc.page_content)


    # =====================================================
    # 11. Reranking
    # =====================================================

    results = rerank_documents(
        standalone_question,
        candidate_results,
        top_k=3
    )

    print(
        "\nNumber of reranked chunks:",
        len(results)
    )


    # =====================================================
    # 12. Display Final Reranked Chunks
    # =====================================================

    print("\n===== Final Reranked Results =====")

    for i, doc in enumerate(results):

        print(
            f"\n===== Reranked Chunk {i + 1} ====="
        )

        print(doc.page_content)


    # =====================================================
    # 13. Format Retrieved Documents
    # =====================================================

    context = format_docs(results)

    print("\n===== Final Context =====")

    print(context)


    # =====================================================
    # 14. Generate Final Answer
    # =====================================================

    answer = generate_answer(
        question,
        context
    )


    # =====================================================
    # 15. Display Final Answer
    # =====================================================

    print("\n===== Final Answer =====")

    print(answer)


    # =====================================================
    # 16. Save Conversation
    # =====================================================

    history.append({
        "question": question,
        "answer": answer
    })