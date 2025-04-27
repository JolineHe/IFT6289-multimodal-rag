import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import random
from app.utils.mongodb import get_collection
import numpy as np

user_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)
rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_random_properties(num_properties: int = 20) -> Tuple[List[Dict[str, Any]], List[int]]:
    collection = get_collection()
    
    # Get total count of documents
    total_docs = collection.count_documents({})
    
    # Generate random indices
    random_indices = random.sample(range(total_docs), min(num_properties, total_docs))
    
    # Get random documents
    properties = []
    ids = []
    for idx in random_indices:
        doc = collection.find().skip(idx).limit(1).next()
        properties.append({
            'id': doc.get('_id', ''),
            'name': doc.get('name', ''),
            'description': doc.get('description', '')          
        })
        ids.append(doc.get('_id', ''))
    return properties, ids


def generate_query_for_property(property_info: Dict[str, str]) -> str:
    """Generate a query for a property using LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates search queries for Airbnb properties."),
        ("human", """
        Based on the following Airbnb property information, generate a query that a potential guest might use to search for this property, such as "I want to stay in a house with a pool, a kitchen, and a balcony ...". 
        This query should be concise and corresponding to the property information, and should not be a question. Focus on the unique aspects mentioned in the description. Query should between 30 and 50 words.
        The query should be in the same language as the description.
        Description: {description}
        
        Query: """)
    ])
    
    chain = prompt | user_llm
    response = chain.invoke({
        "description": property_info['description']
    })
    return response.content.strip()


def generate_queries(properties: List[Dict[str, str]]) -> List[Dict[str, str]]:     
    # Generate queries for each property
    results = []
    for property_info in properties:
        query = generate_query_for_property(property_info)
        results.append({
            'id': property_info['id'],
            'name': property_info['name'],
            'description': property_info['description'],
            'generated_query': query
        })
    return results



def generate_response(query, context):      
    prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an Airbnb listing recommendation system. Please:
                1. Respond in the same language as the user
                2. If the user is asking for property recommendations:
                   - Prioritize results with higher search scores
                   - Include the Airbnb listing URL and image URL
                   - Explain why you chose these properties
                   - Highlight features that match the user's criteria
                3. If the user has provided an image, consider visual similarity in your recommendations
                4. Be friendly and helpful in your responses
                5, answer the question in the same language as the query"""),
                ("human", "Answer this user query: {query} with the following context:\n{context}")
            ])

    formatted_messages = prompt_template.format_messages(query=query, context=context)
    return rag_llm(formatted_messages)  


def check_top_k_positions(arr: np.ndarray, value: int, positions: list[int] = [1, 3, 5, 10]) -> np.ndarray:
    """
    Check if a value is in the top k positions of an array.
    
    Args:
        arr: Input array to search in
        value: Value to search for
        positions: List of k positions to check
        
    Returns:
        np.ndarray: Binary array indicating if value is in top k positions
    """
    results = []
    try:
        index = arr.index(value)
        for pos in positions:
            results.append(1 if index < pos else 0)
    except ValueError:
        results = [0] * len(positions)
    
    return np.array(results)








if __name__ == "__main__":
    results = generate_queries(20)
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Name: {result['name']}")
        print(f"Summary: {result['summary']}")
        print(f"Description: {result['description']}")
        print(f"Query: {result['generated_query']}")
        print("-" * 80) 