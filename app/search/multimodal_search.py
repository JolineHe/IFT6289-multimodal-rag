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
from utils.embedding import get_img_embedding, get_text_embedding
from search.pipelines.multimodal_search_pipelines import pipeline_image_only_search, pipeline_multimodal_search


class MultiModalSearch():
    def __init__(self,collection):
        self.collection = collection

    def do_search(self, query_text, query_img, alpha_text=0.5):
        query_text_embedding, query_img_embedding = None, None
        query_img_embedding = get_img_embedding(query_img)
        if query_text is not None and query_text != '':
            query_text_embedding = get_text_embedding(query_text)
            pipeline = pipeline_multimodal_search(query_text_embedding, query_img_embedding, alpha_text)
        else:
            pipeline = pipeline_image_only_search(query_img_embedding)
        return list(self.collection.aggregate(pipeline))

