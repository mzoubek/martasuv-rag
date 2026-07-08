import os
import argparse
import mimetypes
import base64

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")

    parser.add_argument("--image", type=str, help="Path to an image file")
    parser.add_argument(
        "--query",
        type=str,
        help="Text query which will be rewritten according to the image provided",
    )

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    print(mime)

    image_binary = None
    with open(args.image, "rb") as f:
        image_binary = f.read()

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""

    data_url = f"data:{mime};base64,{base64.b64encode(image_binary).decode()}"
    print(data_url[:50])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    response = client.chat.completions.create(
        model="google/gemma-4-26b-a4b-it:free", messages=messages
    )

    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
