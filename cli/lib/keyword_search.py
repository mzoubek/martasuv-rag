import math
import os
import string
import pickle
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_B,
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH,
    CACHE_DIR,
    load_movies,
    BM25_K1,
    SearchResult,
    format_search_result,
)


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, object] = {}
        self.term_frequencies: defaultdict[int, Counter] = defaultdict(Counter)
        self.doc_lengths = {}
        self.indexpath = os.path.join(CACHE_DIR, "index.pkl")
        self.docmappath = os.path.join(CACHE_DIR, "docmap.pkl")
        self.termfreqpath = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def load(self) -> None:
        try:
            self.index = pickle.load(open(self.indexpath, "rb"))
            self.docmap = pickle.load(open(self.docmappath, "rb"))
            self.term_frequencies = pickle.load(open(self.termfreqpath, "rb"))
            self.doc_lengths = pickle.load(open(self.doc_lengths_path, "rb"))
        except FileNotFoundError:
            raise FileNotFoundError

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            input_text = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, input_text)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        pickle.dump(self.index, open(self.indexpath, "wb"))
        pickle.dump(self.docmap, open(self.docmappath, "wb"))
        pickle.dump(self.term_frequencies, open(self.termfreqpath, "wb"))
        pickle.dump(self.doc_lengths, open(self.doc_lengths_path, "wb"))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        total_tokens = len(tokens)
        self.doc_lengths[doc_id] = total_tokens
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        sum = 0
        for key in self.doc_lengths:
            sum += self.doc_lengths[key]
        return sum / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), set())
        return sorted(doc_ids)

    def __get_docs_len(self) -> int:
        return len(self.docmap)

    def get_tf(self, doc_id: int, term: str) -> int:
        if self.term_frequencies[doc_id][term]:
            return self.term_frequencies[doc_id][term]

        return 0

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = self.__get_docs_len()
        term_match_doc_count = len(self.get_documents(term))

        bm25_idf = math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )
        return bm25_idf

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths[doc_id]
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        saturated_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return saturated_tf

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        query_tokens = tokenize_text(query)

        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results


def tokenize_term(term: str) -> str:
    token = tokenize_text(term)

    if len(token) > 1:
        raise Exception("tokenization split word into two or more")

    return token[0]


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found. Please build it first.")
        exit(1)

    query_tokens = tokenize_text(query)
    seen, result = set(), []
    for token in query_tokens:
        matching_docs = index.get_documents(token)
        for doc_id in matching_docs:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = index.docmap[doc_id]
            result.append(doc)
            if len(result) >= limit:
                return result

    return result


def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found. Please build it first.")
        exit(1)

    scored_docs = index.bm25_search(query, limit)
    print(scored_docs)
    result = []
    for doc in scored_docs:
        document = index.docmap[doc[0]]
        result.append(document)
        result[-1]["bm25"] = doc[1]

    return result


def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found. Please build it first.")
        exit(1)

    token = tokenize_term(term)
    bm25_tf = index.get_bm25_tf(doc_id, token, k1, b)
    return bm25_tf


def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found. Please build it first.")
        exit(1)

    token = tokenize_term(term)
    bm25_idf = index.get_bm25_idf(token)
    return bm25_idf


def count_tf_idf(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found. Please build it first.")
        exit(1)

    term_frequency = index.get_tf(doc_id, term)
    inverse_document_frequency = count_inverse_document_fq(term)

    tf_idf = term_frequency * inverse_document_frequency
    return tf_idf


def count_inverse_document_fq(term: str) -> float:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found. Please build it first.")
        exit(1)

    token = tokenize_term(term)
    term_match_doc_count = index.get_documents(token)
    total_doc_count = index.__get_docs_len()
    idf = math.log((total_doc_count + 1) / (len(term_match_doc_count) + 1))
    return idf


def search_term_freq(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found. Please build it first.")
        exit(1)

    token = tokenize_term(term)

    return index.get_tf(doc_id, token)


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    text_tokens = text.split()

    valid_tokens = []
    for token in text_tokens:
        if token:
            valid_tokens.append(token)

    filtered_tokens = []
    for token in valid_tokens:
        if token not in STOPWORDS:
            filtered_tokens.append(token)

    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in filtered_tokens:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens
