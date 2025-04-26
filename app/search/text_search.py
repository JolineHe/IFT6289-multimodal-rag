class TextSearch():
    def __init__(self, collection) -> None:
        self.collection = collection

    def _build_pipeline(self, query_text: str) -> list[dict]:
        pipeline = [
            {
                "$search": {
                    "index": "full_text_search_index", 
                    "text": {
                        "query": query_text,
                        "path": "description"
                    }
                }
            },
            {
                "$addFields": {
                    "search_score": {"$meta": "searchScore"}
                }
            },
            {
                "$sort": {
                    "search_score": -1
                }
            },
            {"$limit": 10}
        ]
        return pipeline

    def do_search(self, text_query: str) -> list[dict]:
        pipeline = self._build_pipeline(text_query)       
        return self.collection.aggregate(pipeline)