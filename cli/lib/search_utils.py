import os
import json

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP_SIZE = 0
DEFAULT_MAX_CHUNK_SIZE = 4
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75


def load_movies():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_metadata():
    with open("cache/chunk_metadata.json", "r") as f:
        data = json.load(f)
    return data["chunks"]
