import argparse

from lib.multimodal_search import verify_image_embedding, image_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify if image embedding works"
    )
    verify_image_embedding_parser.add_argument(
        "image_path", type=str, help="Path to the image file"
    )

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search the dataset using images"
    )
    image_search_parser.add_argument(
        "image_path", type=str, help="Path to the image file"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            result = image_search(args.image_path)

            for i, doc in enumerate(result, start=1):
                print(f"{i}. {doc['title']} (similarity: {doc['score']:.3f})")
                print(f"   {doc['description'][:100]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
