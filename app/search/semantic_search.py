from app.utils.embedding import get_text_embedding


class SemanticSearch():
    def __init__(self, collection) -> None:
        self.collection = collection

    def _build_pipeline(self, query_embedding: list[float]) -> list[dict]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index_text",
                    "queryVector": query_embedding,
                    "path": "text_embeddings",
                    "numCandidates": 150,
                    "limit": 10,
                    "scoreField": "search_score"
                }
            }, 
            {
                "$addFields": {
                    "search_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        return pipeline

    def do_search(self, text_query: str) -> list[dict]:
        query_embedding = get_text_embedding(text_query)
        pipeline = self._build_pipeline(query_embedding)       
        return self.collection.aggregate(pipeline)