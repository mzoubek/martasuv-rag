import argparse
from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.semantic_search import (
    embed_user_query,
    verify_embeddings,
    verify_model,
    embed_text,
    semantic_search,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    subparsers.add_parser("verify_embeddings", help="Verify that embeddings exist")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Embeds the text and provides results"
    )
    embed_text_parser.add_argument("query", type=str, help="Search query")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embeds user query")
    embed_query_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser(
        "search", help="Semantically search the results"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit of retrieved results",
        default=DEFAULT_SEARCH_LIMIT,
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_user_query(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
