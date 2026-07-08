from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.semantic_search import cosine_similarity
from .search_utils import Movie, load_movies


class MultimodalSearch:
    def __init__(self, documents: list[Movie], model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts: list[str] = []
        for doc in self.documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        img = Image.open(image_path)
        embedding = self.model.encode([img])

        return embedding[0]

    def search_with_image(self, image_path: str):
        img_embedding = self.embed_image(image_path)

        similarities = []
        for idx, doc_embedding in enumerate(self.text_embeddings):
            cos_similarity = cosine_similarity(img_embedding, doc_embedding)
            similarities.append((cos_similarity, self.documents[idx]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for i in range(5):
            results.append(
                {
                    "score": similarities[i][0],
                    "title": similarities[i][1]["title"],
                    "description": similarities[i][1]["description"],
                }
            )

        return results


def verify_image_embedding(image_path: str) -> None:
    documents = load_movies()
    search_instance = MultimodalSearch(documents)
    embedding = search_instance.embed_image(image_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search(image_path: str):
    documents = load_movies()
    search_instance = MultimodalSearch(documents)

    result = search_instance.search_with_image(image_path)
    return result
