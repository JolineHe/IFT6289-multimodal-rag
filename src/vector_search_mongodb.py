from pymongo.operations import SearchIndexModel
from pymongo import MongoClient
import time
import openai
import os
from dotenv import load_dotenv

text_embedding_field_name = "text_embeddings"
vector_search_index_name_text = "vector_index_text"
IMG_EMBED_SIZE = 512
TEXT_EMBED_SIZE = 1536


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class VectorSearchMongoDB:
    def __init__(self, db, collection):
        self.db = db
        self.collection = collection

    def create_vector_search_index(self):
        vector_search_index_model = SearchIndexModel(
            definition={
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        text_embedding_field_name: {
                            "dimensions": TEXT_EMBED_SIZE,
                            "similarity": "cosine",
                            "type": "knnVector",
                        }
                    },
                }
            },
            name=vector_search_index_name_text,
        )

        index_exists = False
        for index in self.collection.list_search_indexes():
            if index['name'] == vector_search_index_name_text:
                index_exists = True
                break

        if not index_exists:
            try:
                result = self.collection.create_search_index(model=vector_search_index_model)
                print("Creating index...")
                time.sleep(20)
                print("Index created successfully:", result)
                print("Wait a few minutes before conducting search with index to ensure index intialization")
            except Exception as e:
                print(f"Error creating vector search index: {str(e)}")
        else:
            print(f"Index '{vector_search_index_name_text}' already exists.")

    def get_embedding(self, text):
        if not text or not isinstance(text, str):
            return None

        try:
            embedding = openai.embeddings.create(
                input=text,
                model="text-embedding-3-small", dimensions=TEXT_EMBED_SIZE).data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error in get_embedding: {e}")
            return None

    def do_vector_search(self, user_query, vector_index="vector_index_text"):
        query_embedding = self.get_embedding(user_query)

        if query_embedding is None:
            return "Invalid query or embedding generation failed."

        vector_search_stage = {
            "$vectorSearch": {
                "index": vector_index,
                "queryVector": query_embedding,
                "path": text_embedding_field_name,
                "numCandidates": 150,
                "limit": 20
            }
        }

        pipeline = [vector_search_stage]

        results = self.collection.aggregate(pipeline)

        explain_query_execution = self.db.command(
            'explain',
            {
                'aggregate': self.collection.name,
                'pipeline': pipeline,
                'cursor': {}
            },
            verbosity='executionStats')

        vector_search_explain = explain_query_execution['stages'][0]['$vectorSearch']
        millis_elapsed = vector_search_explain['explain']['collectors']['allCollectorStats']['millisElapsed']

        print(f"Total time for the execution to complete on the database server: {millis_elapsed} milliseconds")

        return list(results)



if __name__ == "__main__":
    uri = os.getenv('MONGODB_URI')
    client = MongoClient(uri)

    db_name = 'airbnb_dataset'
    collection_name = 'airbnb_embeddings'

    db = client[db_name]
    collection = db[collection_name]

    vector_search_mongodb = VectorSearchMongoDB(db, collection)
    vector_search_mongodb.create_vector_search_index()
    results = vector_search_mongodb.do_vector_search("luxury apartment in the heart of downtown Montreal")
    print(results)