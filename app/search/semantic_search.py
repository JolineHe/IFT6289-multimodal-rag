class SemanticSearch():
    def __init__(self, collection) -> None:
        self.collection = collection

    def _build_pipeline(self, query_embedding: list[float]) -> list[dict]:
        pipeline = [
            {
                    "$vectorSearch": {
                    "index": "vector_index_text",
                    "queryVector": query_embedding,
                    "path": "embeddings",
                    "exact": True,
                    "limit": 5
                    }
            }, {
                    "$project": {
                    "_id": 0,
                    "summary": 1,
                    "listing_url": 1,
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                    }
            }
        ]
        return pipeline

    def do_search(self, text_query: str) -> list[dict]:
        pipeline = self._build_pipeline(text_query)       
        return self.collection.aggregate(pipeline)