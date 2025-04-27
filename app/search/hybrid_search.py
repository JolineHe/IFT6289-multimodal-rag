from app.search.pipelines.hybrid_search_pipeline import build_hybrid_search_stage
from app.utils.embedding import get_text_embedding

class HybridSearch():
    def __init__(self, collection) -> None:
        self.collection = collection

    def _build_pipeline(self, query_vector: list[float], query_text: str) -> list[dict]:
        pipeline = build_hybrid_search_stage(query_vector, query_text)
        # TODO: add additional stages
        return pipeline

    def do_search(self, user_query: str) -> list[dict]:
        query_text = user_query
        query_vector = get_text_embedding(query_text)
        pipeline = self._build_pipeline(query_vector, query_text)       
        return self.collection.aggregate(pipeline)
    
