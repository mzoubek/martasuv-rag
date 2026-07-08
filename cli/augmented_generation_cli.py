import argparse

from lib.augmented_generation import rag_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"- {doc['title']}")
            print()
            print("RAG Response:")
            print(f"{result['answer']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
