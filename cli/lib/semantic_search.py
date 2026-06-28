import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_SIZE,
    load_movies,
)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = defaultdict()

    def generate_embedding(self, text):
        if not text:
            raise ValueError("text is either empty or a whitespace")

        encoded_text = self.model.encode([text])
        return encoded_text[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents

        movie_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_list.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_list, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")

            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity_score, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for i in range(limit):
            results.append(
                {
                    "score": similarities[i][0],
                    "title": similarities[i][1]["title"],
                    "description": similarities[i][1]["description"],
                }
            )

        return results


def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def verify_embeddings() -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:\n")

    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"    {result['description'][:100]}...\n")


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_size: int = DEFAULT_OVERLAP_SIZE,
) -> list[str]:
    words: list[str] = text.split()
    chunks: list[str] = []

    for i in range(0, len(words), chunk_size):
        # On first iteration use first word else grab words to create overlap
        chunk_words = words[i if i == 0 else i - overlap_size : i + chunk_size]
        print(chunk_words)
        chunks.append(" ".join(chunk_words))

    return chunks


def chunk_text(text: str, chunk_size: int, overlap_size: int) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def embed_text(text: str) -> None:
    search_instance = SemanticSearch()
    embedding = None
    try:
        embedding = search_instance.generate_embedding(text)
    except ValueError:
        print(
            "There was an error when generating embedding, the text was empty or whitespace"
        )

    if embedding:
        print(f"Text: {text}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {embedding.shape[0]}")


def embed_user_query(query: str):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
