from pymongo.operations import SearchIndexModel
import time
import openai
import os



text_embedding_field_name = "text_embeddings"
vector_search_index_name_text = "vector_index_text"

openai.api_key = os.environ.get("OPENAI_API_KEY")



def create_vector_search_index(collection, vector_search_index_name_text):
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": { # describes how fields in the database documents are indexed and stored
                "dynamic": True, # automatically index new fields that appear in the document
                "fields": { # properties of the fields that will be indexed.
                    text_embedding_field_name: { 
                        "dimensions": 1536, # size of the vector.
                        "similarity": "cosine", # algorithm used to compute the similarity between vectors
                        "type": "knnVector",
                    }
                },
            }
        },
        name=vector_search_index_name_text, # identifier for the vector search index
    )

    """Create a vector search index in a MongoDB collection."""
    index_exists = False
    for index in collection.list_indexes():
        print(index)
        if index['name'] == vector_search_index_name_text:
            index_exists = True
            break

    if not index_exists:
        try:
            result = collection.create_search_index(model=vector_search_index_model)
            print("Creating index...")
            time.sleep(20)  # Sleep for 20 seconds, adding sleep to ensure vector index has compeleted inital sync before utilization
            print("Index created successfully:", result)
            print("Wait a few minutes before conducting search with index to ensure index intialization")
        except Exception as e:
            print(f"Error creating vector search index: {str(e)}")
    else:
        print(f"Index '{vector_search_index_name_text}' already exists.")


def get_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""

    # Check for valid input
    if not text or not isinstance(text, str):
        return None

    try:
        # Call OpenAI API to get the embedding
        embedding = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small", dimensions=1536).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None


def vector_search(user_query, db, collection, vector_index="vector_index_text"):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    db (MongoClient.database): The database object.
    collection (MongoCollection): The MongoDB collection to search.
    additional_stages (list): Additional aggregation stages to include in the pipeline.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search stage
    vector_search_stage = {
        "$vectorSearch": {
            "index": vector_index, # specifies the index to use for the search
            "queryVector": query_embedding, # the vector representing the query
            "path": text_embedding_field_name, # field in the documents containing the vectors to search against
            "numCandidates": 150, # number of candidate matches to consider
            "limit": 20 # return top 20 matches
        }
    }

    # Define the aggregate pipeline with the vector search stage and additional stages
    pipeline = [vector_search_stage]

    # Execute the search
    results = collection.aggregate(pipeline)

    explain_query_execution = db.command( # sends a database command directly to the MongoDB server
        'explain', { # return information about how MongoDB executes a query or command without actually running it
            'aggregate': collection.name, # specifies the name of the collection on which the aggregation is performed
            'pipeline': pipeline, # the aggregation pipeline to analyze
            'cursor': {} # indicates that default cursor behavior should be used
        }, 
        verbosity='executionStats') # detailed statistics about the execution of each stage of the aggregation pipeline


    vector_search_explain = explain_query_execution['stages'][0]['$vectorSearch']
    millis_elapsed = vector_search_explain['explain']['collectors']['allCollectorStats']['millisElapsed']

    print(f"Total time for the execution to complete on the database server: {millis_elapsed} milliseconds")

    return list(results)