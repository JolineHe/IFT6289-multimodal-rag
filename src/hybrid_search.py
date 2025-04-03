from pipelines.hybrid_search_pipeline import build_hybrid_search_stage
from utils.embedding import get_text_embedding
from utils.mongodb import get_collection

class HybridSearch():
    def __init__(self, collection):
        self.collection = collection

    def _build_pipeline(self, query_vector, query_text):
        pipeline =  build_hybrid_search_stage(query_vector, query_text)
        # TODO: add image search pipeline

        # TODO: add additional stages
        return pipeline
    

    def do_search(self, user_query):
        query_text = user_query
        query_vector = get_text_embedding(query_text)
        pipeline = self._build_pipeline(query_vector, query_text)       
        return self.collection.aggregate(pipeline)
    


if __name__ == "__main__":
    collection = get_collection()
    hybridsearch = HybridSearch(collection)
    results = hybridsearch.do_search("Fully furnished 3+1 flat decorated with vintage style.")
    for result in results:
        print(result)