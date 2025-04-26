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

from utils.embedding import get_img_embedding, get_text_embedding
from app.pipelines.multimodal_pipelines import pipeline_image_only_search, pipeline_multimodal_search

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


class MultiModalSearch():
    def __init__(self,collection):
        self.collection = collection

    def do_search(self, query_text, query_img):
        query_embedding_text, query_embedding_img = None, None
        query_embedding_img = get_img_embedding(query_img)
        if not query_text and query_text != '':
            query_embedding_text = get_text_embedding(query_text)
            pipeline = pipeline_multimodal_search(query_embedding_text, query_embedding_img)
        else:
            pipeline = pipeline_image_only_search(query_embedding_img)
        return list(self.collection.aggregate(pipeline))

