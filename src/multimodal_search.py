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
import openai
import os
from dotenv import load_dotenv

from src.utils.embedding import get_img_embedding, get_text_embedding
from src.pipelines.pipelines_vec import pipeline_vec_single_search, pipeline_vec_multimodal_search, post_boosting_pipeline, post_filter_params_pipeline
from src.utils.check_files import  is_image_file

TEXT_EMBED_FIELD_NAME = "text_embeddings"
IMG_EMBED_FIELD_NAME = "image_embeddings"
vc_index_name_prefix = "vector_index"
ind_suffix = ['text','image']
SCORE_NAME_BASIC = 'search_score'
# RETURN_KEYS = ['_id', 'listing_url', 'name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'property_type', 'room_type', 'bed_type', 'minimum_nights', 'maximum_nights', 'cancellation_policy', 'last_scraped', 'calendar_last_scraped', 'first_review', 'last_review', 'accommodates', 'bedrooms', 'beds', 'number_of_reviews', 'bathrooms', 'amenities', 'price', 'security_deposit', 'cleaning_fee', 'extra_people', 'guests_included', 'images', 'host', 'address', 'availability', 'review_scores', 'reviews']
RETURN_KEYS = ['_id','name','accommodates','address','summary',  'description', 'neighborhood_overview', 'notes', 'images', 'listing_url']

NUM_CANDIDATES = 150
TOP_K = 20


# environment loading
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class MultiSearch_Twopipe:
    '''
    Support multi-modal vector search or single-modal vector search, but each modality is
    searched in a separate pipeline.
    '''
    def __init__(self, collection):
        self.collection = collection

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
        user_query_text = user_query['text']
        user_query_img_path = ''
        if len(user_query) > 1:
            user_query_img_path = user_query['files'][0]
        query_embedding_text = get_text_embedding(user_query_text)
        query_embedding_img = get_img_embedding(user_query_img_path)

        results_text = None
        if not query_embedding_text is None:
            pipeline_text = pipeline_vec_single_search(query_embedding_text,'text')
            # Run the combined pipeline
            results_text = list(self.collection.aggregate(pipeline_text))

        results_image = None
        if query_embedding_img is not None:
            pipeline_image = pipeline_vec_single_search(query_embedding_img,'image')
            results_image = list(self.collection.aggregate(pipeline_image))

        sorted_results = self._reweight_and_merge_results(results_text, results_image,alpha_text)

        return sorted_results

class MultiSearch_Onepipe:
    def __init__(self, collection):
        self.collection = collection

    def do_search(self, user_query, alpha_text=0.5):
        user_query_text = user_query['text']
        user_query_img_path = ''
        if len(user_query) > 1:
            user_query_img_path = user_query['files'][0]
        query_embedding_text = get_text_embedding(user_query_text)
        query_embedding_img = get_img_embedding(user_query_img_path)

        pipeline = pipeline_vec_multimodal_search(
            query_embedding_text, query_embedding_img, alpha_text)
        results = self.collection.aggregate(pipeline)

        return results

class HybridSearch_Onepipe:
    def __init__(self, collection):
        self.collection = collection

    def do_search(self, user_query, alpha_text=0.5):
        # Logic for hybrid search (combining single and multi search)
        user_query_text = user_query['text']
        user_query_img_path = ''
        if len(user_query.get('files')) >= 1:
            user_query_img_path = user_query['files'][0]
        query_embedding_text = get_text_embedding(user_query_text)
        query_embedding_img = get_img_embedding(user_query_img_path) if not user_query_img_path=='' else None

        results = None
        addition_query = {k: v for k, v in user_query.items() if k not in ['text', 'files']}
        pipeline_filter = post_filter_params_pipeline(addition_query)
        pipeline_boost = post_boosting_pipeline()

        if not query_embedding_text is None:
            pipeline_text = pipeline_vec_single_search(query_embedding_text, 'text')
            pipeline_text[0]['$vectorSearch']['filter'] = pipeline_filter
            pipeline = pipeline_text #+ pipeline_boost
            results = list(self.collection.aggregate(pipeline))


        if query_embedding_img is not None:
            pipeline_image = pipeline_vec_single_search(query_embedding_img, 'image')
            pipeline = pipeline_image+pipeline_boost
            results = list(self.collection.aggregate(pipeline))



        return results

class MultiModalSearch():
    def __init__(self,collection):
        self.collection = collection

    def _get_search_engine(self,user_query):
        '''
        If all query vectors exist, in default try multimodal search in one pipeline;
        otherwise try two pipelines if params_others is empty.

        :param user_query: The user query contains the paths of image files and text
        :type user_query: dict
        :param params_others: The dict of other parameters used for hybrid search.
        :type params_others:dict
        :return: None
        :rtype: None
        '''
        user_query_text = user_query['text']
        q_embed_text = get_text_embedding(user_query_text)
        text_flag = q_embed_text is not None

        img_flag = False
        if len(user_query.get('files')) >= 1:
            user_query_img_path = user_query['files'][0]
            img_flag = is_image_file(user_query_img_path)

        param_flag = len(user_query.keys())>2 # more than text and files keys, have other param keys

        if img_flag and text_flag: # all query vectors exist, default try one pipeline
            if not param_flag:
                self.search_type = 'one_pipeline_search'
                self.search_engine = MultiSearch_Onepipe(self.collection)
            else:
                self.search_type = 'one_pipeline_hybridsearch'
                self.search_engine = HybridSearch_Onepipe(self.collection)
        else: # one query vector is missing
            if not param_flag:
                self.search_type = 'two_pipeline_search'
                self.search_engine = MultiSearch_Twopipe(self.collection)
            else:
                self.search_type = 'one_pipeline_hybridsearch'
                self.search_engine = HybridSearch_Onepipe(self.collection)


    def do_search(self, user_query, alpha_text=0.5):
        self._get_search_engine(user_query)
        return self.search_engine.do_search(user_query, alpha_text) #, self.search_type

if __name__ == "__main__":
    from utils.mongodb import get_collection
    collection = get_collection()

    # load an image
    img_path = '../data/image_plateau_montRoyal.png'
    alpha_text = 0.3

    aquery = {
        'text': "I want to stay in a cozy apartment at Plateau Mont Royal",
        'files': [img_path],
        # 'bedrooms': 1,
        # 'address.country': 'Canada',
    }

    vector_search_mongodb = MultiModalSearch(collection)
    results = vector_search_mongodb.do_search(
        aquery, alpha_text=alpha_text)
    # results, search_type = vector_search_mongodb.do_search([
    #     "I want to stay in a cozy apartment at Plateau Mont Royal",
    #     img_path], alpha_text=alpha_text, other_params={}
    # )
    # print(results)
    # # displaying scores and image to evaluate
    """
    To test the search score in seperated text searching or image searching, uncomment below.
    And uncomment 
    "return self.search_engine.do_search(user_query, alpha_text) #, self.search_type"
    """
    # from utils.evl_search_score import evl_search_result
    # evl_search_result(results, img_path, title=f'search_by_multimodal-text{alpha_text}',savepath=f'../data/{search_type}')

    # results, search_type = vector_search_mongodb.do_search([
    #     "recommend a cozy apartment next to metro and similar as the image",
    #     ''], alpha_text=1, other_params={}
    # )
    # # displaying scores and image to evaluate
    # # from utils.evl_search_score import evl_search_result
    #
    # evl_search_result(results, img_path, title='search_by_text',savepath=f'../data/{search_type}')

    # results, search_type  = vector_search_mongodb.do_search([
    #     "",
    #     img_path], alpha_text=0,  other_params={}
    # )
    # # displaying scores and image to evaluate
    # evl_search_result(results, img_path, title='search_by_image',savepath=f'../data/{search_type}')

