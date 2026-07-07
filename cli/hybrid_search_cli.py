import argparse
from lib.search_utils import ALPHA, DEFAULT_SEARCH_LIMIT, K_PARAM
from lib.hybrid_search import (
    normalize_command,
    weighted_search_command,
    rrf_search_command,
)


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

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform Reciprocical ranked fusion search"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        help="K constant which controls how much more weight is given to higher-ranked vs lower-ranked results",
        default=K_PARAM,
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        help="Number of results to return (default=5)",
        default=DEFAULT_SEARCH_LIMIT,
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Retrieved reranking method",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        help="Use LLM as judge to evaluate results",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(
                args.query,
                args.k,
                args.limit,
                args.enhance,
                args.rerank_method,
                args.evaluate,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
