import argparse
from lib.semantic_search import verify_embeddings, verify_model, embed_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    subparsers.add_parser("verify_embeddings", help="Verify that embeddings exist")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Embeds the text and provides results"
    )
    embed_text_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
