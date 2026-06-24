#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    search_command,
    build_command,
    search_term_freq,
    count_inverse_document_fq,
    count_tf_idf,
    bm25_idf_command,
    bm25_tf_command,
    bm25_search_command,
)

from lib.search_utils import BM25_K1, BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    term_frequencies = subparsers.add_parser("tf", help="Count term frequencies")
    term_frequencies.add_argument("doc_id", type=int, help="Document ID")
    term_frequencies.add_argument("term", type=str, help="Search term")

    inverse_document_frequency = subparsers.add_parser(
        "idf", help="Count inverse document frequencies"
    )
    inverse_document_frequency.add_argument("term", type=str, help="Search term")

    term_fq_inverse_doc_fq = subparsers.add_parser(
        "tfidf", help="Count term TF-IDF for specific doc_id and term"
    )
    term_fq_inverse_doc_fq.add_argument("doc_id", type=int, help="Document ID")
    term_fq_inverse_doc_fq.add_argument("term", type=str, help="Search term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            results = search_command(args.query)
            print(f"Searching for: {args.query}")
            for i, res in enumerate(results, start=1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            print(f"Searching for doc_id, term: {args.doc_id}, {args.term}")
            count = search_term_freq(args.doc_id, args.term)
            print(count)
        case "idf":
            print(f"Counting IDF for term: {args.term}")
            idf = count_inverse_document_fq(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            print(f"Counting TF-IDF from doc_id: {args.doc_id} of term: {args.term}")
            tf_idf = count_tf_idf(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            print(f"Counting BM25 IDF for term: {args.term}")
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            print(
                f"Counting BM25 TF from doc_id: {args.doc_id} of term: {args.term} with k1 of {args.k1}"
            )
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}"
            )
        case "bm25search":
            print(f"Searching using BM25 for query: {args.query}")
            bm25_results = bm25_search_command(args.query)
            for i, res in enumerate(bm25_results, start=1):
                print(f"{i}. ({res['id']}) {res['title']} - {res['bm25']:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
