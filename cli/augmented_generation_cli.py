import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.augmented_generation import rag, summarize, citations


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument(
        "--limit",
        type=int,
        help="Limit of retrieved results",
        default=DEFAULT_SEARCH_LIMIT,
    )

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform search + generate summarization"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        help="Limit of retrieved results",
        default=DEFAULT_SEARCH_LIMIT,
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Provides answer to searched query with sources"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument(
        "--limit",
        type=int,
        help="Limit of retrieved results",
        default=DEFAULT_SEARCH_LIMIT,
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag(args.query, args.limit)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"- {doc['title']}")
            print()
            print("RAG Response:")
            print(f"{result['answer']}")
        case "summarize":
            result = summarize(args.query, args.limit)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"- {doc['title']}")
            print()
            print("LLM Summary:")
            print(f"{result['answer']}")
        case "citations":
            result = citations(args.query, args.limit)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"- {doc['title']}")
            print()
            print("LLM Answer:")
            print(f"{result['answer']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
