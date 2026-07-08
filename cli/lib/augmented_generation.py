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


def generate_answer(search_results, query, method: str, limit=5):
    context: str = ""
    for doc in search_results[:limit]:
        context += f"{doc['title']}: {doc['description']}\n\n"

    prompt: str = ""
    match method:
        case "rag":
            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {context}

            Answer:"""
        case "summarize":
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Search results:
            {context}

            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
        case "citations":
            prompt = f"""Answer the query below and give information based on the provided documents.

            The answer should be tailored to users of Hoopla, a movie streaming service.
            If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

            Query: {query}

            Documents:
            {context}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources in the format [1], [2], etc. when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the provided documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
        case "question":
            prompt = f"""Answer the following question based on the provided documents.

            Question: {query}

            Documents:
            {context}

            General instructions:
            - Answer directly and concisely
            - Use only information from the documents
            - If the answer isn't in the documents, say "I don't have enough information"
            - Cite sources when possible

            Guidance on types of questions:
            - Factual questions: Provide a direct answer
            - Analytical questions: Compare and contrast information from the documents
            - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

            Answer:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def rag(method: str, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    search_instance = HybridSearch(documents)

    search_results = search_instance.rrf_search(
        query, k=RRF_K, limit=5 * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {"query": query, "search_results": [], "error": "No results found"}

    answer = generate_answer(search_results, query, method, limit)

    return {"query": query, "search_results": search_results[:limit], "answer": answer}
