from typing import List, Optional, Dict, Any
from pymongo import MongoClient
from pydantic import BaseModel
from data_models import Address, Image_Describ
from hybrid_search import HybridSearch
from multimodal_search import MultiModalSearch
from utils.logger import LOG
from utils.session_history import get_session_history
import pandas as pd
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage

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


class RagAgent:
    def __init__(self, collection):
        self.collection = collection
        self.hybrid_search = HybridSearch(collection)
        self.multimodal_search = MultiModalSearch(collection)
        self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def retrieve_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge from MongoDB based on the query.
        Args:
            query: Dictionary containing 'text' and optionally 'files'
        Returns:
            List of search results with relevant information
        """
        if len(query.get('files', [])) == 0:
            get_knowledge = self.hybrid_search.do_search(query['text'])
        else:
            get_knowledge = self.multimodal_search.do_search([query['text'], query['files'][0]])

        # Check if there are any results
        if not get_knowledge:
            return []

        # Convert search results into a list of SearchResultItem models
        search_results_models = [
            SearchResultItem(**result)
            for result in get_knowledge
        ]

        # Prepare the context for LangChain
        context = [{
            'content': f"""
            Name: {row.name}
            Description: {row.description}
            Summary: {row.summary}
            Neighborhood: {row.neighborhood_overview}
            Notes: {row.notes}
            Search Score: {row.search_score}
            Image URL: {row.images}
            """,
            'score': row.search_score
        } for row in search_results_models]

        return context

    def response_to_user(self, query: Dict[str, Any], session_id: str = "default") -> str:
        """
        Generate a response to the user query using LLM and session history.
        Args:
            query: Dictionary containing 'text' and optionally 'files'
            session_id: Unique identifier for the conversation session
        Returns:
            Generated response string
        """
        # Get or create session history
        history = get_session_history(session_id)

        # First, determine if the query is about property recommendations
        query_type_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a query classifier. Your task is to determine if the user is asking about Airbnb property recommendations. Respond with only 'yes' or 'no'."),
            ("human", "Is this query about Airbnb property recommendations? Query: {query}")
        ])
        
        query_type_response = self.chat(query_type_prompt.format_messages(query=query['text']))
        is_property_query = query_type_response.content.lower().strip() == 'yes'
        has_image = len(query.get('files', [])) > 0

        # If it's a property query or has an image, retrieve knowledge
        context = []
        if is_property_query or has_image:
            context = self.retrieve_knowledge(query)

        # Create different prompt templates based on the query type
        if is_property_query or has_image:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an Airbnb listing recommendation system. Please:
                1. Respond in the same language as the user
                2. If the user is asking for property recommendations:
                   - Prioritize results with higher search scores
                   - Include the Airbnb listing URL and image URL
                   - Explain why you chose these properties
                   - Highlight features that match the user's criteria
                3. If the user has provided an image, consider visual similarity in your recommendations
                4. Be friendly and helpful in your responses"""),
                *history.messages,
                ("human", "Answer this user query: {query} with the following context:\n{context}")
            ])
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an Airbnb customer service assistant. Please:
                1. Respond in the same language as the user
                2. Be friendly and helpful
                3. If you don't know the answer, say so politely
                4. Keep responses concise and relevant"""),
                *history.messages,
                ("human", "Answer this user query: {query}")
            ])
        
        # Format the prompt with the actual values
        if is_property_query or has_image:
            formatted_messages = prompt_template.format_messages(
                query=query,
                context=context if context else "No specific property information available."
            )
        else:
            formatted_messages = prompt_template.format_messages(
                query=query
            )

        # Get response from LLM
        response = self.chat(formatted_messages)
        system_response = response.content

        # Add messages to history
        history.add_user_message(str(query))
        history.add_ai_message(system_response)

        LOG.info(f"- User Question:\n{query}\n")
        LOG.info(f"- System Response:\n{system_response}\n")

        return system_response


if __name__ == "__main__":
    uri = os.getenv('MONGODB_URI')
    client = MongoClient(uri)

    db_name = 'airbnb_dataset'
    collection_name = 'airbnb_embeddings'

    db = client[db_name]
    collection = db[collection_name]

    vector_search = MultiModalSearch(db, collection)
    rag_agent = RagAgent(db, collection, vector_search)
    # load an image
    img_path = './data/image1.png'

    query_text = """
    I want to stay in a place that's warm and friendly, 
    and not too far from resturants, can you recommend a place that is similar as the image I provide? 
    Include a reason as to why you've chosen your selection. Also give me the airbnb link and image link
    """
    rag_agent.handle_user_query([query_text, img_path])