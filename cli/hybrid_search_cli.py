import argparse
from lib.search_utils import ALPHA, DEFAULT_SEARCH_LIMIT
from lib.hybrid_search import normalize_command, weighted_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize the provided list of scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="List of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Perform weighted hybrid search",
    )
    weighted_search_parser.add_argument("query", type=str, help="Searched query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
        default=ALPHA,
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        help="Number of results to return (default=5)",
        default=DEFAULT_SEARCH_LIMIT,
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
