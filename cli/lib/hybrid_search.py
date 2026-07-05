import os
import time
import json

from dotenv import load_dotenv
from openai import OpenAI

from .search_utils import Movie, load_movies

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")


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
                    "description": res["document"][:100],
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
        keyword_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        document_mapping = {}

        for rank, res in enumerate(semantic_results, start=1):
            document = document_mapping.get(res["id"])
            semantic_rank = rrf_score(rank, k)
            if not document:
                document_mapping[res["id"]] = {
                    "title": res["title"],
                    "description": res["document"],
                    "semantic_rank": semantic_rank,
                    "keyword_rank": 0.0,
                }
                continue

            if document["semantic_rank"] < semantic_rank:
                document["semantic_rank"] = semantic_rank

        #
        #   Keyword results addition
        #
        for rank, res in enumerate(keyword_results, start=1):
            document = document_mapping.get(res["id"])
            keyword_rank = rrf_score(rank, k)
            if not document:
                document_mapping[res["id"]] = {
                    "title": res["title"],
                    "description": res["document"][:100],
                    "semantic_rank": 0.0,
                    "keyword_rank": keyword_rank,
                }
                continue

            if document["keyword_rank"] < keyword_rank:
                document["keyword_rank"] = keyword_rank

        for doc in document_mapping.values():
            doc["rrf_score"] = doc["semantic_rank"] + doc["keyword_rank"]

        result = sorted(
            document_mapping.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        return result[:limit]


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


#
#   WARNING: duplication of code
#
def enhance_spell(query: str) -> str | None:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f"""Fix any spelling errors in the user-provided movie search query below.
                            Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                            Preserve punctuation and capitalization unless a change is required for a typo fix.
                            If there are no spelling errors, or if you're unsure, output the original query unchanged.
                            Output only the final query text, nothing else.
                            User query: "{query}"
                            """,
            }
        ],
    )
    enhanced_query: str | None = response.choices[0].message.content
    return enhanced_query


def enhance_rewrite(query: str) -> str | None:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                                Consider:
                                - Common movie knowledge (famous actors, popular films)
                                - Genre conventions (horror = scary, animation = cartoon)
                                - Keep the rewritten query concise (under 10 words)
                                - It should be a Google-style search query, specific enough to yield relevant results
                                - Don't use boolean logic

                                Examples:
                                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                                If you cannot improve the query, output the original unchanged.
                                Output only the rewritten query text, nothing else.

                                User query: "{query}"
                                """,
            }
        ],
    )
    enhanced_query: str | None = response.choices[0].message.content
    return enhanced_query


def enhance_expand(query: str) -> str | None:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f"""Expand the user-provided movie search query below with related terms.

                                Add synonyms and related concepts that might appear in movie descriptions.
                                Keep expansions relevant and focused.
                                Output only the additional terms; they will be appended to the original query.

                                Examples:
                                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                                - "action movie with bear" -> "action thriller bear chase fight adventure"
                                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                                User query: "{query}"
                                """,
            }
        ],
    )
    enhanced_query: str | None = response.choices[0].message.content
    return enhanced_query


def individual_reranking(query: str, results: list[dict]) -> list[dict]:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    for doc in results:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": f"""Rate how well this movie matches the search query.

                                Query: "{query}"
                                Movie: {doc.get("title", "")} - {doc.get("document", "")}

                                Consider:
                                - Direct relevance to query
                                - User intent (what they're looking for)
                                - Content appropriateness

                                Rate 0-10 (10 = perfect match).
                                Output ONLY the number in your response, no other text or explanation.

                                Score:""",
                }
            ],
        )

        if response.choices[0].message.content:
            doc["rerank_score"] = float(response.choices[0].message.content)
        else:
            doc["rerank_score"] = 0.0

        time.sleep(3)

    rerank_result = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return rerank_result


def batch_reranking(query: str, results: list[dict]) -> list[dict]:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    for doc in results:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": f"""Rank the movies listed below by relevance to the following search query.

                                Query: "{query}"

                                Movies:
                                {results}

                                Return the movie IDs in order of relevance, best match first.

                                Your response must be a raw JSON array of integers.
                                Do not wrap the JSON in Markdown. Do not use a ```json code block.
                                Do not include any explanatory text.

                                For example:
                                [75, 12, 34, 2, 1]

                                Ranking:""",
                }
            ],
        )

        ranking_list = json.loads(response.choices[0].message.content)

    return ranked_result


def rrf_search_command(
    query: str, k: int, limit: int, enhance: str, rerank_method: str
):
    documents = load_movies()
    search_instance = HybridSearch(documents)

    enhanced_query: str | None = None
    match enhance:
        case "spell":
            enhanced_query = enhance_spell(query)
        case "rewrite":
            enhanced_query = enhance_rewrite(query)
        case "expand":
            enhanced_query = enhance_expand(query)

    if enhance:
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
    final_query = enhanced_query if enhanced_query else query

    if rerank_method:
        # NOTE: nice magic number probably add constant dumbass
        limit *= 5
    results = search_instance.rrf_search(final_query, k, limit)

    match rerank_method:
        case "individual":
            results = individual_reranking(query, results)
        case "batch":
            results = batch_reranking(query, results)

    for i, res in enumerate(results, start=1):
        print(f"{i}. {res['title']}")
        if rerank_method:
            print(f"  Re-rank Score: {res['rerank_score']:.3f}/10")
        print(f"  RRF Score: {res['rrf_score']:.4f}")
        print(
            f"  BM25 Rank: {res['keyword_rank']:.4f}, Semantic Rank: {res['semantic_rank']:.4f}"
        )
        print(f"  {res['description']}")


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
