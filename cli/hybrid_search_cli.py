import argparse
from lib.hybrid_search import normalize_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize the provided list of scores"
    )
    normalize_parser.add_argument(
        "list_of_scores", type=float, nargs="*", help="List of scores to normalize"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.list_of_scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
