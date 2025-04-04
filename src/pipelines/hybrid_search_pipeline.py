from typing import List, Dict, Any



def build_hybrid_search_stage(
    query_vector: List[float],
    query_text: str = "Fully furnished 3+1 flat decorated with vintage style.",
    vector_weight: float = 0.7,
    full_text_weight: float = 0.3,
    num_candidates: int = 100,
    limit: int = 20,
    final_limit: int = 10) -> List[Dict[str, Any]]:
    """Performs hybrid search combining vector and full-text search results.

    This pipeline executes both vector similarity search and full-text search on MongoDB collections,
    combining the results using Reciprocal Rank Fusion (RRF). The vector search finds semantically 
    similar documents based on embeddings, while full-text search matches text patterns.

    Args:
        query_vector (List[float]): Vector embedding of the search query
        vector_weight (float, optional): Weight given to vector search results. Defaults to 0.5
        full_text_weight (float, optional): Weight given to full-text search results. Defaults to 0.5
        query_text (str, optional): Text query for full-text search. Defaults to "star wars"
        num_candidates (int, optional): Number of candidates to consider in vector search. Defaults to 100
        limit (int, optional): Maximum results to return from each search type. Defaults to 20
        final_limit (int, optional): Final number of results after combining both searches. Defaults to 10

    Returns:
        List[Dict[str, Any]]: Combined and ranked search results, each containing document metadata
    """
    VECTOR_INDEX_NAME = "vector_index_text"
    VECTOR_QUERY_PATH = "text_embeddings"
    FULL_TEXT_INDEX_NAME = "full_text_search_index"
    FULL_TEXT_QUERY_PATH = "description"
    COLLECTION_NAME = "airbnb_embeddings"
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": VECTOR_QUERY_PATH,
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit
            }
        },
        {
            "$group": {
                "_id": None,
                "docs": {"$push": "$$ROOT"}
            }
        },
        {
            "$unwind": {
                "path": "$docs",
                "includeArrayIndex": "rank"
            }
        },
        {
            "$addFields": {
                "vs_score": {
                    "$multiply": [
                        vector_weight,
                        {
                            "$divide": [
                                1.0,
                                {
                                    "$add": ["$rank", 60]
                                }
                            ]
                        }
                    ]
                }
            }
        },
        {
            "$project": {
                "vs_score": 1,
                "_id": "$docs._id",
                "name": "$docs.name"
            }
        },
        {
            "$unionWith": {
                "coll": COLLECTION_NAME,
                "pipeline": [
                    {
                        "$search": {
                            "index": FULL_TEXT_INDEX_NAME,
                            "phrase": {
                                "query": query_text,
                                "path": FULL_TEXT_QUERY_PATH
                            }
                        }
                    },
                    {"$limit": 20},
                    {
                        "$group": {
                            "_id": None,
                            "docs": {"$push": "$$ROOT"}
                        }
                    },
                    {
                        "$unwind": {
                            "path": "$docs",
                            "includeArrayIndex": "rank"
                        }
                    },
                    {
                        "$addFields": {
                            "fts_score": {
                                "$multiply": [
                                    full_text_weight,
                                    {
                                        "$divide": [
                                            1.0,
                                            {
                                                "$add": ["$rank", 60]
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "fts_score": 1,
                            "_id": "$docs._id",
                            "name": "$docs.name"
                        }
                    }
                ]
            }
        },
        {
            "$group": {
                "_id": "$_id",
                "name": {"$first": "$name"},
                "vs_score": {"$max": "$vs_score"},
                "fts_score": {"$max": "$fts_score"}
            }
        },
        {
            "$project": {
                "_id": 1,
                "name": 1,
                "vs_score": {"$ifNull": ["$vs_score", 0]},
                "fts_score": {"$ifNull": ["$fts_score", 0]}
            }
        },
        {
            "$project": {
                "score": {"$add": ["$fts_score", "$vs_score"]},
                "_id": 1,
                "name": 1,
                "vs_score": 1,
                "fts_score": 1
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": final_limit}
    ]
    return pipeline
