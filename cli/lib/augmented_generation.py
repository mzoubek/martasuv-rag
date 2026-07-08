import os

from dotenv import load_dotenv
from openai import OpenAI

from .search_utils import DEFAULT_SEARCH_LIMIT, RRF_K, SEARCH_MULTIPLIER, load_movies
from .hybrid_search import HybridSearch


load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"


def generate_answer(search_results, query, limit=5):
    context: str = ""
    for doc in search_results[:limit]:
        context += f"{doc['title']}: {doc['description']}\n\n"

    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {context}

    Answer:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def rag_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    search_instance = HybridSearch(documents)

    search_results = search_instance.rrf_search(
        query, k=RRF_K, limit=5 * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {"query": query, "search_results": [], "error": "No results found"}

    answer = generate_answer(search_results, query, limit)

    return {"query": query, "search_results": search_results[:limit], "answer": answer}
