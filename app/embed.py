from .config import Settings


class Embedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class NoopEmbedder(Embedder):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class LocalSentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def get_embedder(settings: Settings) -> Embedder:
    if settings.embed_provider.lower() == "local":
        return LocalSentenceTransformerEmbedder(settings.embed_model_name)
    return NoopEmbedder()
