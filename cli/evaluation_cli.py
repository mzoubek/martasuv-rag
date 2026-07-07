import argparse

from lib.evaluation import evaluation_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    evaluation_command(limit)


# run evaluation logic here

if __name__ == "__main__":
    main()
