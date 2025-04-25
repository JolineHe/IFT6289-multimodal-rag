import os
from pymongo import MongoClient
import openai

TEXT_EMBED_MODEL = "text-embedding-3-small"
TEXT_EMBED_SIZE = 1536

def get_collection():
    uri = os.getenv('MONGODB_URI')
    client = MongoClient(uri)
    db_name = 'airbnb_dataset'  # Change this to your actual database name
    collection_name = 'airbnb_embeddings'  # Change this to your actual collection name
    return client[db_name][collection_name]



def get_text_embedding(text):
        if not text or not isinstance(text, str):
            print("text is not a string")
            return None
        try:
            embedding = openai.embeddings.create(
                input=text,
                model=TEXT_EMBED_MODEL, dimensions=TEXT_EMBED_SIZE).data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error in get_text_embedding: {e}")
            return None
