import os

from .search_utils import Movie, load_movies

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.indexpath):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        keyword_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        min_kw_score, max_kw_score = (
            min([item["score"] for item in keyword_results]),
            max([item["score"] for item in keyword_results]),
        )
        min_sem_score, max_sem_score = (
            min([item["score"] for item in semantic_results]),
            max([item["score"] for item in semantic_results]),
        )

        document_mapping = {}
        #
        #   Semantic results addition
        #
        for res in semantic_results:
            document = document_mapping.get(res["id"])
            normalized_score = normalize(res["score"], min_sem_score, max_sem_score)
            if not document:
                document_mapping[res["id"]] = {
                    "title": res["title"],
                    "description": res["document"],
                    "semantic_score": normalized_score,
                    "keyword_score": 0.0,
                }
                continue

            if document["semantic_score"] < normalized_score:
                document["semantic_score"] = normalized_score

        #
        #   Keyword results addition
        #
        for res in keyword_results:
            document = document_mapping.get(res["id"])
            normalized_score = normalize(res["score"], min_kw_score, max_kw_score)
            if not document:
                document_mapping[res["id"]] = {
                    "title": res["title"],
                    "description": res["description"][:100],
                    "semantic_score": 0.0,
                    "keyword_score": normalized_score,
                }
                continue

            keyword_score = document.get("keyword_score")
            if not keyword_score or keyword_score < normalized_score:
                document.update({"keyword_score": normalized_score})

        #
        #   Hybrid score addition to final doc mapping
        #
        for value in document_mapping.values():
            keyword_score = value["keyword_score"]
            semantic_score = value["semantic_score"]
            value["hybrid_score"] = hybrid_score(keyword_score, semantic_score, alpha)

        result = sorted(
            document_mapping.values(), key=lambda x: x["hybrid_score"], reverse=True
        )

        return result[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def weighted_search_command(query: str, alpha: float, limit: int = 5) -> None:
    documents = load_movies()
    search_instance = HybridSearch(documents)

    results = search_instance.weighted_search(query, alpha, limit)
    # TODO: Adjust once works correctly
    for i, res in enumerate(results, start=1):
        print(f"{i}. {res['title']}")
        print(f"  Hybrid Score: {res['hybrid_score']:.4f}")
        print(
            f"  BM25: {res['keyword_score']:.4f}, Semantic: {res['semantic_score']:.4f}"
        )
        print(f"  {res['description']}")


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize(score: float, min: float, max: float) -> float:
    return (score - min) / (max - min)


def normalize_command(scores: list[float]) -> None:
    if not scores:
        return

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        print("* 1.0")
        return

    for score in scores:
        score = normalize(score, min_score, max_score)
        print(f"* {score:.4f}")
