from pymongo.operations import SearchIndexModel
import time
from database.mongodb_utils import get_collection
# settings of database and index
TEXT_EMBED_FIELD_NAME = "text_embeddings"
IMG_EMBED_FIELD_NAME = "image_embeddings"
vc_index_name_prefix = "vector_index"
ind_suffix = ['text','image']
VC_INDEX_DICT = {k: f"{vc_index_name_prefix}_{k}" for k in ind_suffix}
SCORE_NAME_BASIC = 'search_score'
RETURN_KEYS = ['_id', 'listing_url', 'name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'property_type', 'room_type', 'bed_type', 'minimum_nights', 'maximum_nights', 'cancellation_policy', 'last_scraped', 'calendar_last_scraped', 'first_review', 'last_review', 'accommodates', 'bedrooms', 'beds', 'number_of_reviews', 'bathrooms', 'amenities', 'price', 'security_deposit', 'cleaning_fee', 'extra_people', 'guests_included', 'images', 'host', 'address', 'availability', 'review_scores', 'reviews', 'text_embeddings', 'image_embeddings']
NUM_CANDIDATES = 150
TOP_K = 20
TEXT_EMBED_SIZE = 1536
IMG_EMBED_SIZE = 512



def if_index_exist(collection, index_name):
    index_exists = False
    for index in collection.list_search_indexes():
        if index['name'] == index_name:
            index_exists = True
            print(f"Index of '{index_name}' already exists.")
            break
    return index_exists


def create_vector_search_index_model(index_name, type='text'):
    index_name = VC_INDEX_DICT[type]
    field_name = TEXT_EMBED_FIELD_NAME if type == 'text' else IMG_EMBED_FIELD_NAME
    score_name = SCORE_NAME_BASIC
    if type == 'text' or type == 'image':
        score_name = f'{type}_{SCORE_NAME_BASIC}'
    embed_size = IMG_EMBED_SIZE if type=='image' else TEXT_EMBED_SIZE
    vector_search_index_model = SearchIndexModel(
        definition={
            "type": "vectorSearch",
            "mappings": {
                "dynamic": True,
                "fields": {
                    field_name: {
                        "dimensions": embed_size,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },

                },
            },
            "options": {
                "scoreField": score_name  # This ensures the score is returned
            }
        },
        name=index_name,
    )
    return vector_search_index_model, index_name

##create full text search index
def create_search_index_model(index_name, field_name):
    search_index_model = SearchIndexModel(
        definition={
            "type": "search",
            "mappings": {
                "dynamic": False,
                "fields": {
                    field_name: [{
                        "type": "string",
                    }]
                }
            }
        },
        name=index_name,
    )
    return search_index_model, index_name



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
    index_model, index_name = create_search_index_model("full_text_search_index", "description")
    if not if_index_exist(collection, index_name):
        create_indexes(index_model, collection)