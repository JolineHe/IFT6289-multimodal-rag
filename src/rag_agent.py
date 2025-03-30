from typing import List, Optional
from pymongo import MongoClient
from pydantic import BaseModel
from data_models import Address
from vector_search_mongodb import VectorSearchMongoDB
import pandas as pd
import openai
import os


class SearchResultItem(BaseModel):
    name: str
    accommodates: Optional[int] = None
    address: Address
    summary: Optional[str] = None
    description: Optional[str] = None
    neighborhood_overview: Optional[str] = None
    notes: Optional[str] = None


class RagAgent:
    def __init__(self, db, collection, vector_search):
        self.db = db
        self.collection = collection
        self.vector_search = vector_search

    def handle_user_query(self, query):
        # Assuming vector_search returns a list of dictionaries with keys 'title' and 'plot'
        get_knowledge = self.vector_search.do_vector_search(query)

        # Check if there are any results
        if not get_knowledge:
            return "No results found.", "No source information available."

        # Convert search results into a list of SearchResultItem models
        search_results_models = [
            SearchResultItem(**result)
            for result in get_knowledge
        ]

    # Convert search results into a DataFrame for better rendering in Jupyter
    search_results_df = pd.DataFrame([item.dict() for item in search_results_models])

    # Generate system response using OpenAI's completion
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": "You are a airbnb listing recommendation system."},
            {
                "role": "user", 
                "content": f"Answer this user query: {query} with the following context:\n{search_results_df}"
            }
        ]
    )

    system_response = completion.choices[0].message.content

    # Print User Question, System Response, and Source Information
    print(f"- User Question:\n{query}\n")
    print(f"- System Response:\n{system_response}\n")

    # Display the DataFrame as an HTML table
    display(HTML(search_results_df.to_html()))

    # Return structured response and source info as a string
    return system_response



if __name__ == "__main__":
    uri = os.getenv('MONGODB_URI')
    client = MongoClient(uri)

    db_name = 'airbnb_dataset'
    collection_name = 'airbnb_embeddings'

    db = client[db_name]
    collection = db[collection_name]

    vector_search = VectorSearchMongoDB(db, collection)
    rag_agent = RagAgent(db, collection, vector_search)

    query = """
    I want to stay in a place that's warm and friendly, 
    and not too far from resturants, can you recommend a place? 
    Include a reason as to why you've chosen your selection.
    """
    rag_agent.handle_user_query(query)
