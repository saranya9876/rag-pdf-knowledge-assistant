def evaluate_retrieval(query, docs):
    score = len(docs)
    print(f"Retrieved chunks: {score}")
    return score