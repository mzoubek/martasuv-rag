import os
import re
import json
from collections import defaultdict
from typing import TypedDict
from sentence_transformers import SentenceTransformer
import numpy as np

from .search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_SIZE,
    load_metadata,
    load_movies,
)


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[ChunkMetadata] = []

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        chunks: list[str] = []
        chunks_metadata: list[dict] = []
        for i, doc in enumerate(documents):
            if not doc["description"]:
                continue

            chunk_sentences: list[str] = semantic_chunking(doc["description"], 4, 1)
            for idx, chunk in enumerate(chunk_sentences):
                chunks.append(chunk)
                chunks_metadata.append(
                    {
                        "movie_idx": i,
                        "chunk_idx": idx,
                        "total_chunks": len(chunk_sentences),
                    }
                )

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata

        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(
                {"chunks": chunks_metadata, "total_chunks": len(chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            self.chunk_metadata = load_metadata()

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )
        query_embedding = self.generate_embedding(query)

        chunk_scores: list[dict] = []
        for i, chunk in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(query_embedding, chunk)
            chunk_metadata = self.chunk_metadata[i]
            chunk_scores.append(
                {
                    "chunk_idx": chunk_metadata["chunk_idx"],
                    "movie_idx": chunk_metadata["movie_idx"],
                    "score": similarity_score,
                }
            )

        movie_scores: dict[int, float] = {}
        for chunk in chunk_scores:
            movie_idx_score = movie_scores.get(chunk["movie_idx"])
            if not movie_idx_score or movie_idx_score < chunk["score"]:
                movie_idx = chunk["movie_idx"]
                score = chunk["score"]
                movie_scores.update({movie_idx: score})

        sorted_best_chunks = sorted(
            movie_scores.items(), key=lambda x: x[1], reverse=True
        )

        results = []
        for movie_idx, score in sorted_best_chunks[:limit]:
            if movie_idx is None:
                continue

            document = self.documents[movie_idx]
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "document": document["description"],
                    "score": score,
                }
            )

        return results


def search_chunked(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    search_instance = ChunkedSemanticSearch()

    search_instance.load_or_create_chunk_embeddings(documents)

    results = search_instance.search_chunks(query, limit)

    for i, res in enumerate(results, start=1):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")


def embed_chunks() -> None:
    documents = load_movies()
    search_instance = ChunkedSemanticSearch()

    embeddings = search_instance.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


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
    i = 0

    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap_size

    return chunks


def semantic_chunking(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_SIZE,
):
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and sentences[0].endswith((".", "!", "?")):
        sentences = [text]

    chunks: list[str] = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            chunk_sentence = chunk_sentence.strip()
            if chunk_sentence:
                cleaned_sentences.append(chunk_sentence)

        if not cleaned_sentences:
            i += max_chunk_size - overlap
            continue

        chunk = " ".join(chunk_sentences)
        chunks.append(chunk)
        i += max_chunk_size - overlap

    return chunks


def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_SIZE,
):
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_size: int = DEFAULT_OVERLAP_SIZE,
) -> None:
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
