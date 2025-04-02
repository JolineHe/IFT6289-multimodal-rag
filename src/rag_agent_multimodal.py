from typing import List, Optional
from pymongo import MongoClient
from pydantic import BaseModel
from data_models import Address, Image_Describ
from src.multimodal_vector_search_mongodb import MultiModalVectorSearchMongoDB
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
    images: Image_Describ
    search_score: Optional[float] = None

def prompt_topk_search_score(top_results):
    top_results_text = "\n\n".join(
        [f"Result {i+1} (Score: {doc['search_score']}): "
         f"{doc['text']}" for i, doc in enumerate(top_results)]
    )

    prompt_text = f"""Here are the top search results, ranked by similarity score:
    
    {top_results_text}
    
    Please synthesize the best answer using the most relevant information.
    """

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
        search_results_df = pd.DataFrame([item.model_dump() for item in search_results_models])
        df_img = pd.json_normalize(search_results_df['images'])
        search_results_df['images'] = df_img['picture_url'].values.tolist()

        search_results_df = search_results_df.sort_values(by="search_score", ascending=False)

        # Generate system response using OpenAI's completion
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a airbnb listing recommendation system."},
                {
                    "role": "system",
                    "content": f"Please use the result with high search_score at {search_results_df.search_score}.",},
                {
                    "role": "system",
                    "content": f"Please search the image link from \n{search_results_df.images}"},
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

        # Return structured response and source info as a string
        return system_response


if __name__ == "__main__":
    uri = os.getenv('MONGODB_URI')
    client = MongoClient(uri)

    db_name = 'airbnb_dataset'
    collection_name = 'airbnb_embeddings'

    db = client[db_name]
    collection = db[collection_name]

    vector_search = MultiModalVectorSearchMongoDB(db, collection)
    rag_agent = RagAgent(db, collection, vector_search)
    # load an image
    img_path = './data/image1.png'

    query_text = """
    I want to stay in a place that's warm and friendly, 
    and not too far from resturants, can you recommend a place that is similar as the image I provide? 
    Include a reason as to why you've chosen your selection. Also give me the airbnb link and image link
    """
    rag_agent.handle_user_query([query_text])