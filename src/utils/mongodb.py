import os
from pymongo import MongoClient

def get_collection():
    uri = os.getenv('MONGODB_URI')
    client = MongoClient(uri)
    db_name = 'airbnb_dataset'  # Change this to your actual database name
    collection_name = 'airbnb_embeddings'  # Change this to your actual collection name
    return client[db_name][collection_name]