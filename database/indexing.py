from pymongo.operations import SearchIndexModel
import time
from mongodb_utils import get_collection


def if_index_exist(collection, index_name):
    index_exists = False
    for index in collection.list_search_indexes():
        if index['name'] == index_name:
            index_exists = True
            print(f"Index of '{index_name}' already exists.")
            break
    return index_exists


def create_text_vector_index_model():
    vector_search_index_model = SearchIndexModel(
        definition={
            "type": "vectorSearch",
            "mappings": {
                "dynamic": True,
                "fields": {
                    "description_embedding": {
                        "dimensions": 1536,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },

                },
            },
            "options": {
                "scoreField": "text_search_score"  # This ensures the score is returned
            }
        },
        name="vector_index_text",
    )
    return vector_search_index_model, "vector_index_text"



def create_image_vector_index_model():
    vector_search_index_model = SearchIndexModel(
        definition={
            "type": "vectorSearch",
            "mappings": {
                "dynamic": True,
                "fields": {
                    "image_embeddings": {
                        "dimensions": 512,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },

                },
            },
            "options": {
                "scoreField": "image_search_score"  # This ensures the score is returned
            }
        },
        name=index_name,
    )
    return vector_search_index_model, index_name




##create full text search index
def create_full_text_index_model():
    search_index_model = SearchIndexModel(
        definition={
            "type": "search",
            "mappings": {
                "dynamic": False,
                "fields": {
                    "description": [{
                        "type": "string",
                    }]
                }
            }
        },
        name="full_text_search_index",
    )
    return search_index_model, "full_text_search_index"



def create_indexes(index_model, collection):
    try:
        result = collection.create_search_index(model=index_model)
        print("Creating index...")
        time.sleep(20)
        print("Index created successfully:", result)
        print("Wait a few minutes before conducting search with index to ensure index intialization")
    except Exception as e:
        print(f"Error creating vector search index: {str(e)}")
    


if __name__ == "__main__":
    collection = get_collection()
    index_model, index_name = create_text_vector_index_model()
    if not if_index_exist(collection, index_name):
        create_indexes(index_model, collection)