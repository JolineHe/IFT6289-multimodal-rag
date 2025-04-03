#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Miao at 2025-03-23
'''
@File   : multimodal_vector_search_mongodb
@Time   : 2025-03-23 3:48 p.m.
@Author : Miao
@Project: IFT6289-multimodal-rag 
@Desc   : Please enter here
'''
from pymongo import MongoClient
import openai
import os
from dotenv import load_dotenv
from utils.logger import LOG
from utils.embedding import get_img_embedding, get_text_embedding


TEXT_EMBED_FIELD_NAME = "text_embeddings"
IMG_EMBED_FIELD_NAME = "image_embeddings"
vc_index_name_prefix = "vector_index"
ind_suffix = ['text','image']
SCORE_NAME_BASIC = 'search_score'
RETURN_KEYS = ['_id', 'listing_url', 'name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'property_type', 'room_type', 'bed_type', 'minimum_nights', 'maximum_nights', 'cancellation_policy', 'last_scraped', 'calendar_last_scraped', 'first_review', 'last_review', 'accommodates', 'bedrooms', 'beds', 'number_of_reviews', 'bathrooms', 'amenities', 'price', 'security_deposit', 'cleaning_fee', 'extra_people', 'guests_included', 'images', 'host', 'address', 'availability', 'review_scores', 'reviews', 'text_embeddings', 'image_embeddings']
NUM_CANDIDATES = 150
TOP_K = 20


# environment loading
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class MultiModalSearch:
    def __init__(self, collection):
        self.collection = collection    

    def _build_pipeline(self, embedding, type='text'):
        assert type in ['text', 'image']
        ind_name = f"{vc_index_name_prefix}_{type}"

        score_name = f'{type}_{SCORE_NAME_BASIC}'
        embed_name = IMG_EMBED_FIELD_NAME if type=='image' else TEXT_EMBED_FIELD_NAME

        pipeline = []
        pipeline.append({
            "$vectorSearch": {
                "index": ind_name,
                "path": embed_name, #"image_embeddings",
                "queryVector": embedding,
                "numCandidates": NUM_CANDIDATES,
                "limit": TOP_K,
                "scoreField": score_name
            }
        })
        out_template = {key: 1 for key in RETURN_KEYS + [score_name]}
        out_template[score_name] = {"$meta": "vectorSearchScore"}
        pipeline.append({
            "$project": out_template
        })

        return pipeline

    def _reweight_and_merge_results(self, text_results, image_results,
                                   alpha_text=0.5):
        """
        Merges image and text search results based on ObjectId, reweights the text and image scores,
        and returns the top K results sorted by their combined score.

        :param image_results: List of search results with image scores ('_id' and 'image_search_score').
        :param text_results: List of search results with text scores ('_id' and 'text_search_score').
        :param alpha_text: Weight for text similarity (0-1). Higher = more text importance.
        :param top_k: Number of top results to return.
        :return: List of merged and reweighted results.
        """

        if image_results is None:
            return text_results[:TOP_K]

        if text_results is None:
            return image_results[:TOP_K]

        # Convert text results to a dictionary for fast lookup by _id
        text_results_dict = {str(result["_id"]): result for result in text_results}

        # List to hold merged and reweighted results
        merged_results = []

        img_sname, text_sname = f'image_{SCORE_NAME_BASIC}', f'text_{SCORE_NAME_BASIC}'

        for image_result in image_results:
            image_id = str(image_result["_id"])
            image_search_score = image_result.get(img_sname, 0)

            # Try to find the corresponding text result by _id
            text_result = text_results_dict.get(image_id)

            # If a text result exists, get its score; if not, set text score to 0
            text_search_score = text_result.get(text_sname, 0) if text_result else 0

            # Compute the reweighted search score
            search_score = (
                    text_search_score * alpha_text +
                    image_search_score * (1 - alpha_text)
            )

            # Merge the results
            merged_result = {
                "_id": image_result["_id"],  # Use image_result as the base
                img_sname: image_search_score,
                text_sname: text_search_score,
                SCORE_NAME_BASIC: search_score,
            }
            merge_rest = {k: image_result.get(k) for k in RETURN_KEYS}

            aResult = {**merged_result, **merge_rest}

            merged_results.append(aResult)

        # Sort the merged results by the combined search score (descending)
        merged_results.sort(key=lambda x: x[SCORE_NAME_BASIC], reverse=True)

        # Return the top K results
        return merged_results[:TOP_K]

    def do_search(self, user_query, alpha_text=0.5):
        user_query_text = user_query[0]
        user_query_img_path = ''
        if len(user_query) > 1:
            user_query_img_path = user_query[1]
        query_embedding_text = get_text_embedding(user_query_text)
        query_embedding_img = get_img_embedding(user_query_img_path)

        results_text = None
        if not query_embedding_text is None:
            pipeline_text = self._build_pipeline(query_embedding_text,'text')
            # Run the combined pipeline
            results_text = list(self.collection.aggregate(pipeline_text))

        results_image = None
        if query_embedding_img is not None:
            pipeline_image = self._build_pipeline(query_embedding_img,'image')
            results_image = list(self.collection.aggregate(pipeline_image))

        sorted_results = self._reweight_and_merge_results(results_text, results_image)

        return sorted_results



if __name__ == "__main__":
    uri = os.getenv('MONGODB_URI')
    client = MongoClient(uri)

    db_name = 'airbnb_dataset'
    collection_name = 'airbnb_embeddings'

    db = client[db_name]
    collection = db[collection_name]

    # load an image
    img_path = './data/image1.png'
    alpha_text = 0.3

    vector_search_mongodb = MultiModalSearch(db, collection)
    results = vector_search_mongodb.do_vector_search([
        "recommend a cozy apartment next to metro and similar as the image",
        img_path], alpha_text=alpha_text
    )
    # # displaying scores and image to evaluate
    # from evl.evl_search_score import evl_search_result
    # evl_search_result(results, img_path, title=f'search_by_multimodal-text{alpha_text}')
    """
        To test the search score in seperated text searching or image searching, uncomment below.
    """
    # results = vector_search_mongodb.do_vector_search([
    #     "recommend a cozy apartment next to metro and similar as the image",
    #     ''], alpha_text=1
    # )
    # # displaying scores and image to evaluate
    # from evl.evl_search_score import evl_search_result
    #
    # evl_search_result(results, img_path, title='search_by_text')

    # results = vector_search_mongodb.do_vector_search([
    #     "",
    #     img_path], alpha_text=0
    # )
    # # displaying scores and image to evaluate
    # evl_search_result(results, img_path, title='search_by_image')

