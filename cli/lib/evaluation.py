from .search_utils import load_golden_dataset, load_movies
from .hybrid_search import HybridSearch


def evaluation(limit: int):
    documents = load_movies()
    search_instance = HybridSearch(documents)
    test_cases = load_golden_dataset()

    results = []
    for test in test_cases:
        query = test["query"]

        # WARNING: possibly get rid of magic number, we don't like them
        search_results = search_instance.rrf_search(query, 60, limit)

        relevant_retrieved = 0
        total_retrieved = len(search_results)
        titles_retrieved = []
        for res in search_results:
            titles_retrieved.append(res["title"])

            if res["title"] in test["relevant_docs"]:
                relevant_retrieved += 1

        titles_retrieved_str = ", ".join(titles_retrieved)
        relevant_docs_str = ", ".join(test["relevant_docs"])

        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / len(test["relevant_docs"])
        results.append(
            {
                "query": query,
                "precision": precision,
                "recall": recall,
                "f1_score": 2 * (precision * recall) / (precision + recall),
                "retrieved": titles_retrieved_str,
                "relevant": relevant_docs_str,
            }
        )

    return results


def evaluation_command(limit: int):
    results = evaluation(limit)

    if not results:
        raise ValueError("Results from evaluation are non-existing")

    print(f"k={limit}\n")
    for res in results:
        print(f"- Query: {res['query']}")
        print(f"  - Precision@{limit}: {res['precision']:.4f}")
        print(f"  - Recall@{limit}: {res['recall']:.4f}")
        print(f"  - F1 Score: {res['f1_score']:.4f}")
        print(f"  - Retrieved: {res['retrieved']}")
        print(f"  - Relevant: {res['relevant']}\n")
