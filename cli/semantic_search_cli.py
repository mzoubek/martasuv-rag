import argparse
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
)
from lib.semantic_search import (
    embed_user_query,
    semantic_chunk,
    verify_embeddings,
    verify_model,
    embed_text,
    semantic_search,
    chunk_text,
    embed_chunks,
    search_chunked,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    subparsers.add_parser("verify_embeddings", help="Verify that embeddings exist")
    subparsers.add_parser("embed_chunks", help="Embeds chunks and saves to files")

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

    chunk_parser = subparsers.add_parser(
        "chunk", help="Provides example of how chunking works"
    )
    chunk_parser.add_argument("text", type=str, help="Text which you want to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        help="Settable chunk size, default is 200",
        default=DEFAULT_CHUNK_SIZE,
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        help="Settable overlapping chunk size, default is 0 == off",
        default=DEFAULT_OVERLAP_SIZE,
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Provides example of how semantic chunking works"
    )
    semantic_chunk_parser.add_argument(
        "text", type=str, help="Text which you want to chunk"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        help="Settable max chunk size",
        default=DEFAULT_MAX_CHUNK_SIZE,
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        help="Settable overlapping chunk size, default is 0 == off",
        default=DEFAULT_OVERLAP_SIZE,
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search using semantic search on chunked embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Searched query")
    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        help="Limit of retrieved results",
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
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
