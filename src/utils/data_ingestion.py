import os
from datasets import load_dataset
from bson import json_util
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()  

def ingest_data():
    # MongoDB Atlas URI and client setup
    uri = os.getenv('MONGODB_URI')
    client = MongoClient(uri)

    # Change to the appropriate database and collection names
    db_name = 'airbnb_dataset'  # Change this to your actual database name
    collection_name = 'airbnb_embeddings'  # Change this to your actual collection name

    collection = client[db_name][collection_name]

    #NOTE: https://huggingface.co/datasets/MongoDB/airbnb_embeddings
    #NOTE: This dataset contains several records with datapoint representing an airbnb listing.
    dataset = load_dataset("MongoDB/airbnb_embeddings")

    insert_data = []

    # Iterate through the dataset and prepare the documents for insertion
    # The script below ingests 1000 records into the database at a time
    for item in dataset['train']:
        # Convert the dataset item to MongoDB document format
        doc_item = json_util.loads(json_util.dumps(item))
        if doc_item['description'] is not None:
            insert_data.append(doc_item)

        # Insert in batches of 1000 documents
        if len(insert_data) == 1000:
            collection.insert_many(insert_data)
            print("1000 records ingested")
            insert_data = []

    # Insert any remaining documents
    if len(insert_data) > 0:
        collection.insert_many(insert_data)
        print("{} records ingested".format(len(insert_data)))

    print("All records ingested successfully!")


def delete_empty_descriptions():
    """
    Delete documents where description field is empty from the MongoDB collection.
    
    Args:
        collection: MongoDB collection object
    """
    uri = os.getenv('MONGODB_URI')
    client = MongoClient(uri)

    # Change to the appropriate database and collection names
    db_name = 'airbnb_dataset'  # Change this to your actual database name
    collection_name = 'airbnb_embeddings'  # Change this to your actual collection name

    collection = client[db_name][collection_name]

    # Delete documents where description is None, empty string, or just whitespace
    result = collection.delete_many({
        "$or": [
            {"description": None},
            {"description": ""},
            {"description": {"$regex": r"^\s*$"}}
        ]
    })
    
    print(f"Deleted {result.deleted_count} documents with empty descriptions")



if __name__ == "__main__":
    # ingest_data()
    delete_empty_descriptions()
    