from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from utils.algo_apiori import findApiroriWithRules, findFrequentItemSet, generateAssociationRules
import uvicorn
import time
import worker
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from utils.hybrid_recommend.collaborative_filtering import CollaborativeFiltering
from utils.hybrid_recommend.content_base import ContentBasedRecommender
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
import numpy as np


@staticmethod
def combine_recommendations(cf_recommendations, cb_recommendations, cf_weight=0.5, cb_weight=0.5):
    """
    Combine recommendations from collaborative filtering and content-based filtering.
    
    Args:
        cf_recommendations: List of tuples (movieId, rating) from collaborative filtering
        cb_recommendations: List of tuples (movieId, rating) from content-based filtering
        cf_weight: Weight for collaborative filtering predictions (default: 0.5)
        cb_weight: Weight for content-based predictions (default: 0.5)
        
    Returns:
        List of tuples (movieId, combined_rating) sorted by combined rating
    """
    # Create dictionaries for easy lookup
    cf_dict = {movie_id: rating for movie_id, rating in cf_recommendations}
    cb_dict = {movie_id: rating for movie_id, rating in cb_recommendations}
    
    # Get all unique movie IDs
    all_movie_ids = set(cf_dict.keys()) | set(cb_dict.keys())
    
    # Combine predictions
    combined_predictions = []
    for movie_id in all_movie_ids:
        cf_rating = cf_dict.get(movie_id, 0)
        cb_rating = cb_dict.get(movie_id, 0)
        
        # Calculate weighted average
        combined_rating = (cf_rating * cf_weight + cb_rating * cb_weight)
        combined_predictions.append((movie_id, combined_rating))
    
    # Sort by combined rating (second value in tuple) in descending order
    combined_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 5 items
    top_5 = combined_predictions[:5]
    
    # Print top 5 with their ratings
    print("\nTop 5 Combined Recommendations:")
    for movie_id, rating in top_5:
        print(f"Movie ID: {movie_id}, Combined Rating: {rating:.4f}")
    
    return combined_predictions



ratings_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\movies_dataset\ratings_small.csv')
movies_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\movies_dataset\movies_with_keywords.csv')
userId = 1
similarity_matrix = np.load(r"C:\Users\Admin\Documents\Last semester\Data Mining Exercise\utils\hybrid_recommend\weight\similar_item_item_matrix.npy")
cf = CollaborativeFiltering(n_neighbors=5, movies_df=movies_df, ratings_df=ratings_df)
cb = ContentBasedRecommender(movies_df=movies_df, ratings_df=ratings_df)
recommendations, top_recommendations = cf.recommend_items(userId, n_recommendations=5)
cb_recommendations, cb_top_recommendations = cb.recommend_items(userId, similarity_matrix=similarity_matrix)
# Combine recommendations from both methods
combined_recommendations = combine_recommendations(cf_recommendations=recommendations, cb_recommendations=cb_recommendations, cf_weight=0.5, cb_weight=0.5)