from datasets import load_dataset
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from pymongo.mongo_client import MongoClient
import os

from data_models import *


# NOTE: https://huggingface.co/datasets/MongoDB/airbnb_embeddings
# NOTE: This dataset contains several records with datapoint representing an airbnb listing.



MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DATABASE = os.environ.get("MONGO_DATABASE")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION")
DATASET = "MongoDB/airbnb_embeddings"



def load_dataset():
    dataset = load_dataset(DATASET, streaming=True, split="train")
    dataset_df = pd.DataFrame(dataset)
    records = dataset_df.to_dict(orient='records')

    # To handle catch `NaT` values
    for record in records:
        for key, value in record.items():
            # Check if the value is list-like; if so, process each element.
            if isinstance(value, list):
                processed_list = [None if pd.isnull(v) else v for v in value]
                record[key] = processed_list
            # For scalar values, continue as before.
            else:
                if pd.isnull(value):
                    record[key] = None

    try:
        listings = [Listing(**record).dict() for record in records]
        # Get an overview of a single datapoint
        print(listings[0].keys())
    except ValidationError as e:
        print(e)
    return listings



def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri, appname="devrel.deeplearningai.lesson1.python")
    print("Connection to MongoDB successful")
    return client



def ingest_data(listings):
    if not MONGO_URI:
        print("MONGO_URI not set in environment variables")
    mongo_client = get_mongo_client(MONGO_URI)

    # Pymongo client of database and collection
    db = mongo_client.get_database(MONGO_DATABASE)
    collection = db.get_collection(MONGO_COLLECTION)

    collection.delete_many({})
    collection.insert_many(listings)
    print("Data ingestion into MongoDB completed")



if __name__ == "__main__":
    listings = load_dataset()
    ingest_data(listings)