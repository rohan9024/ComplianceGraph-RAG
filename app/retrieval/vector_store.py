from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from app.config import settings

class QdrantRetriever:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection = settings.QDRANT_COLLECTION
        # Loads supported FastEmbed model
        self.embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
        self._init_collection()

    def _init_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            # Auto-detect vector dimension from the loaded model
            sample_vec = list(self.embedding_model.embed(["test"]))[0]
            vector_size = len(sample_vec)

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )

    def index_documents(self, documents: List[Dict[str, Any]]):
        if not documents:
            return

        texts = [doc["text"] for doc in documents]
        embeddings = list(self.embedding_model.embed(texts))

        points = []
        for i, (doc, vec) in enumerate(zip(documents, embeddings)):
            points.append(
                models.PointStruct(
                    id=i,
                    vector=vec.tolist(),
                    payload={
                        "chunk_id": doc["id"],
                        "text": doc["text"],
                        "ticker": doc.get("ticker", "UNKNOWN"),
                        "filing": doc.get("filing", "UNKNOWN"),
                        "source": doc.get("source", "Kaggle-10K")
                    }
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        query_vec = list(self.embedding_model.embed([query]))[0].tolist()
        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=top_k
        ).points

        return [
            {
                "id": hit.payload.get("chunk_id"),
                "text": hit.payload.get("text"),
                "ticker": hit.payload.get("ticker"),
                "filing": hit.payload.get("filing"),
                "score": hit.score
            }
            for hit in hits
        ]