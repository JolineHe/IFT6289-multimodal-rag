from typing import List, Optional
from pydantic import BaseModel
from data_models import Address, ImageDescrib
from hybrid_search import HybridSearch
from multimodal_search import MultiModalSearch
from utils.logger import LOG
import pandas as pd
import openai
import os

ALPHA_TEXT = 0.4
class SearchResultItem(BaseModel):
    name: str
    accommodates: Optional[int] = None
    address: Address
    summary: Optional[str] = None
    description: Optional[str] = None
    neighborhood_overview: Optional[str] = None
    notes: Optional[str] = None
    images: ImageDescrib
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
    def __init__(self, collection):
        self.collection = collection
        self.hybrid_search = HybridSearch(collection)
        self.multimodal_search = MultiModalSearch(collection)

    def handle_user_query(self, query,other_params=dict()):
        # Todo
        # How to split each input case to one specific search engine, need to redefine clearly.
        if len(query.get('files') ) == 0: # input contains texts, no image, no params with come here.
            get_knowledge = self.hybrid_search.do_search(query['text'])
        else: # if the input contains (texts, images) and or params, will use this part
            get_knowledge = self.multimodal_search.do_search(
                [query['text'], query['files'][0]],
                alpha_text=ALPHA_TEXT,
                other_params=other_params
            )

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
        LOG.info(f"- User Question:\n{query}\n")
        LOG.info(f"- System Response:\n{system_response}\n")


        # Return structured response and source info as a string
        return system_response


if __name__ == "__main__":
    from utils.mongodb import get_collection

    collection = get_collection()
    rag_agent = RagAgent(collection)
    # load an image
    img_path = '../data/image_plateau_montRoyal.png'

    query_text = """
    I want to stay in a place that's warm and friendly, 
    and not too far from resturants, can you recommend a place that is similar as the image I provide? 
    Include a reason as to why you've chosen your selection. Also give me the airbnb link and image link
    """
    rag_agent.handle_user_query([query_text, img_path])