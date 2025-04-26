#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Miao at 2025-04-03
'''
@File   : pipes
@Time   : 2025-04-03 10:27 p.m.
@Author : Miao
@Project: IFT6289-multimodal-rag 
@Desc   : Please enter here
'''
from typing import List, Dict, Any

TEXT_EMBED_FIELD_NAME = "text_embeddings"
IMG_EMBED_FIELD_NAME = "image_embeddings"
vc_index_name_prefix = "vector_index"
ind_suffix = ['text','image']
SCORE_NAME_BASIC = 'search_score'
# RETURN_KEYS = ['_id', 'listing_url', 'name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'property_type', 'room_type', 'bed_type', 'minimum_nights', 'maximum_nights', 'cancellation_policy', 'last_scraped', 'calendar_last_scraped', 'first_review', 'last_review', 'accommodates', 'bedrooms', 'beds', 'number_of_reviews', 'bathrooms', 'amenities', 'price', 'security_deposit', 'cleaning_fee', 'extra_people', 'guests_included', 'images', 'host', 'address', 'availability', 'review_scores', 'reviews']
RETURN_KEYS = ['name','accommodates','address','summary',  'description', 'neighborhood_overview', 'notes', 'images', 'listing_url']
COLLECTION_NAME = "airbnb_embeddings"
NUM_CANDIDATES = 150
TOP_K = 20

def pipeline_image_only_search(query_vector: List[float]) -> List[Dict[str, Any]]:
    ind_name = "vector_index_image"
    score_name = "image_search_score"
    embed_name = IMG_EMBED_FIELD_NAME

    pipeline = []
    pipeline.append({
        "$vectorSearch": {
            "index": ind_name,
            "path": embed_name, 
            "queryVector": query_vector,
            "numCandidates": NUM_CANDIDATES,
            "limit": TOP_K,
            "scoreField": score_name
        }
    })
    out_template = {key: 1 for key in RETURN_KEYS + [score_name]}
    out_template[score_name] = {"$meta": "vectorSearchScore"}
    out_template["_id"] = 1
    pipeline.append({
        "$project": out_template
    })
    return pipeline

def pipeline_multimodal_search(
    query_vector_text: List[float],
    query_vector_image: List[float],
    text_weight: float = 0.4) -> List[Dict[str, Any]]:
    """Performs hybrid search combining vector and full-text search results.

    This pipeline executes both vector similarity search and full-text search on MongoDB collections,
    combining the results using Reciprocal Rank Fusion (RRF). The vector search finds semantically
    similar documents based on embeddings, while full-text search matches text patterns.

    Args:
        query_vector_text (List[float]): Vector embedding of the queried text
        query_vector_image (List[float]): Vector embedding of the queried image
        text_weight (float): Text embedding weight, between [0,1]
    Returns:
        List[Dict[str, Any]]: Combined and ranked search results, each containing document metadata
    """
    return_dict = {i: f"$docs.{i}" for i in RETURN_KEYS}
    select_dict = {i: {"$first": f"${i}"} for i in RETURN_KEYS}
    flag_return_dict = {i: 1 for i in RETURN_KEYS}

    # embed_text, embed_img = embeddings_list
    alpha_img, alpha_text = 1 - text_weight, text_weight

    ind_name_text, ind_name_img = f"{vc_index_name_prefix}_text", f"{vc_index_name_prefix}_image"
    score_name_text, score_name_img = f"text_{SCORE_NAME_BASIC}", f"image_{SCORE_NAME_BASIC}"
    embed_name_text, embed_name_img = TEXT_EMBED_FIELD_NAME, IMG_EMBED_FIELD_NAME

    # pipeline = []
    k_param_text = 5  # Adjust this value to balance text and image scores
    k_param_img = 5

    pipeline = [
        {
            "$vectorSearch": {
                "index": ind_name_text,
                "path": embed_name_text,
                "queryVector": query_vector_text,
                "numCandidates": NUM_CANDIDATES,
                "limit": TOP_K
            }
        }, {
            "$group": {
                "_id": None,
                "docs": {"$push": "$$ROOT"}
            }
        }, {
            "$unwind": {
                "path": "$docs",
                "includeArrayIndex": "rank"
            }
        }, {
            "$addFields": {
                score_name_text: {
                    "$multiply": [
                        alpha_text, {
                            "$divide": [
                                1.0, {
                                    "$add": ["$rank", k_param_text]
                                }
                            ]
                        }
                    ]
                }
            }
        }, {
            "$project": {
                score_name_text: 1,
                "_id": "$docs._id", # note is "$docs._id" not "$id"
                **return_dict,
            }
        }, # --------- Union with Vector Search 2: IMAGE ---------
        {
            "$unionWith": {
                "coll": COLLECTION_NAME,
                "pipeline": [
                    {
                        "$vectorSearch": {
                            "index": ind_name_img,
                            "path": embed_name_img,
                            "queryVector": query_vector_image,
                            "numCandidates": NUM_CANDIDATES,
                            "limit": TOP_K
                        }
                    }, {
                        "$limit": TOP_K
                    }, {
                        "$group": {
                            "_id": None,
                            "docs": {"$push": "$$ROOT"}
                        }
                    }, {
                        "$unwind": {
                            "path": "$docs",
                            "includeArrayIndex": "rank"
                        }
                    }, {
                        "$addFields": {
                            score_name_img: {
                                "$multiply": [
                                    alpha_img, {
                                        "$divide": [
                                            1.0, {
                                                "$add": ["$rank", k_param_img]
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            score_name_img: 1,
                            "_id": "$docs._id",
                            **return_dict,
                        }
                    }
                ]
            }
        }, {
            "$group": {
                "_id": "$_id",
                **select_dict,
                score_name_img: {"$max": f"${score_name_img}"},
                score_name_text: {"$max": f"${score_name_text}"}
            }
        }, {
            "$project": {
                "_id": 1,
                **flag_return_dict,
                score_name_img: {"$ifNull": [f"${score_name_img}", 0]},
                score_name_text: {"$ifNull": [f"${score_name_text}", 0]}
            }
        }, {
            "$project": {
                SCORE_NAME_BASIC: {"$add": [f"${score_name_img}", f"${score_name_text}"]},
                "_id": 1,
                **flag_return_dict,
                score_name_img: 1,
                score_name_text: 1
            }
        },
        {"$sort": {SCORE_NAME_BASIC: -1}},
        {"$limit": TOP_K}
    ]
    return pipeline

