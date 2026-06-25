import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np

from .search_utils import load_movies


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


def embed_text(text: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    vec1_len = len(vec1)
    vec2_len = len(vec2)
    if vec1_len != vec2_len:
        raise ValueError("provided vectors with different lengths")

    result_vec: list[float] = []
    for i in range(vec1_len):
        result_vec.append(vec1[i] + vec2[i])

    return result_vec


def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    vec1_len = len(vec1)
    vec2_len = len(vec2)
    if vec1_len != vec2_len:
        raise ValueError("provided vectors with different lengths")

    result_vec: list[float] = []
    for i in range(vec1_len):
        result_vec.append(vec1[i] - vec2[i])

    return result_vec
