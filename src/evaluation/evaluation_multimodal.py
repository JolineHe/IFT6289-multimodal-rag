import sys
import os
import requests

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import random
from src.utils.mongodb import get_collection
from src.utils.embedding import get_img_embedding
import numpy as np

user_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_random_img_properties(num_properties: int = 20) -> Tuple[List[Dict[str, Any]], List[int]]:
    collection = get_collection()
    
    # Get total count of documents
    total_docs = collection.count_documents({})
    
    # Generate random indices
    random_indices = random.sample(range(total_docs), min(num_properties*5, total_docs))
    
    # Get random documents
    properties = []
    ids = []
    cnt = 0
    for idx in random_indices:
        if cnt == num_properties: # when hit the required number of properties
            break
        doc = collection.find().skip(idx).limit(1).next()
        if doc.get('image_embeddings', '') is None or doc.get('images').get('picture_url')=='':
            continue
        properties.append({
            'id': doc.get('_id', ''),
            'name': doc.get('name', ''),
            'description': doc.get('description', ''),
            'image_url': doc.get('images').get('picture_url'),
        })
        cnt += 1
        ids.append(doc.get('_id', ''))
    return properties, ids


def generate_query_for_property(property_info: Dict[str, str]) -> str:
    """Generate a query for a property using LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates search queries for Airbnb properties."),
        ("human", """
        Based on the following Airbnb property information, generate a query that a potential guest 
        might use to search for this property, such as "I want to stay in a house with a pool, a kitchen, and a balcony ...". 
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
            'generated_query': query,
            'image_path': property_info['image_url'],
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

class Gen_MultimodalEvlDataset:
    '''
    Rewrite the pure text query or ground truth to include image information,
    where the image information is created by Image Caption Model.
    '''
    def __init__(self):
        # Load model and processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def _gen_caption(self,img_path):
        # Load image (you can also use a local path)
        try:
            image = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
            # Preprocess and generate
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except:
            return ""


    def gen_combined_query(self,query): # for each sample
        image_caption = self._gen_caption(query['files'][0])
        combined_query = f"{query['text']}. \n --Image looks like: {image_caption}"
        return combined_query

    def gen_combined_retrieved(self,retrieved_list): # for the whole list
        results = []
        for i in retrieved_list:
            descrip = i['description'][0]
            image_path = i['images'][0]['picture_url']
            image_caption = self._gen_caption(image_path)
            combined_query = f"{descrip}. \n --Image looks like: {image_caption}"
            results.append(combined_query)

        return results

    def gen_combined_gt(self,ground_list):
        results = []
        for i in ground_list:
            desciption = i['description']
            image_path = i['image_url']
            image_caption = self._gen_caption(image_path)
            combined_query = f"{desciption}. \n --Image looks like: {image_caption}"
            results.append(combined_query)

        return results




if __name__ == "__main__":
    random_properties, random_property_ids = get_random_img_properties(5)
    results = generate_queries(random_properties)
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Name: {result['name']}")
        # print(f"Summary: {result['summary']}")
        print(f"Description: {result['description']}")
        print(f"Query: {result['generated_query']}")
        print(f"Image Path: {result['image_path']}")
        print("-" * 80)

    data_generator = Gen_MultimodalEvlDataset()

    print(data_generator.gen_combined_query({'text': 'find an appartment',
                                             'files': ['https://a0.muscache.com/im/pictures/6ae01e26-1aec-40cd-ad92-7dc0f188df3d.jpg?aki_policy=large']}))