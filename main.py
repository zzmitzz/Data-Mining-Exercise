from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from utils.algo_apiori import findApiroriWithRules, findFrequentItemSet, generateAssociationRules
from utils.algo_fpgrowth import fp_growth, generate_association_rules_from_fp
import uvicorn
import time
import worker
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from utils.hybrid_recommend import content_base
from utils.hybrid_recommend.recommending import HybridRecommender
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
from utils.apriori_hashtree import apriori_hash_tree

app = FastAPI()
origins = [
    "http://localhost:5173",
]

# Create a thread pool executor
thread_pool = ThreadPoolExecutor(max_workers=10)

# Thread-local storage for dataframes
thread_local = threading.local()

def get_ratings_df():
    if not hasattr(thread_local, 'ratings_df'):
        thread_local.ratings_df = pd.read_csv(ratingTrans)
    return thread_local.ratings_df

def get_movies_df():
    if not hasattr(thread_local, 'movies_df'):
        thread_local.movies_df = pd.read_csv(movies)
    return thread_local.movies_df

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,        # allow cookies, Authorization headers
    allow_methods=["*"],           # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],           # allow all headers
)
class CustomDataset(BaseModel):
    data: List[List[str]]
    min_support: Optional[float] = 0.2
    min_confidence: Optional[float] = 0.5
    min_rating_filter: Optional[float] = 3.5

@app.post("/apriori/custom")
async def run_apriori_custom(dataset: CustomDataset):
    start_time = time.time()
    try:
        # Convert the incoming 2D list into the format expected by the algorithm
        itemset = set()
        for transaction in dataset.data:
            for item in transaction:
                itemset.add(tuple([item]))
        
        freq_itemsets, support_dict = findFrequentItemSet(
            initItemSet=itemset,
            transactions=dataset.data,
            minSupport=dataset.min_support
        )
        transformed_itemsets = {}
        for k, itemsets in freq_itemsets.items():
            transformed_itemsets[k] = [
                {
                    'items': itemset,
                    'support': support_dict[itemset]
                }
                for itemset in itemsets
            ]
        rules = generateAssociationRules(
            frequentItemSets=freq_itemsets,
            support_dict=support_dict,
            minConfidence=dataset.min_confidence
        )
        
        return {
            "frequent_itemsets": transformed_itemsets,
            "rules": rules,
            "time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/apriori/default")
async def run_apriori_default(
    min_support: Optional[float] = 0.2,
    min_confidence: Optional[float] = 0.5,
    min_rating_filter: Optional[float] = 3.5
):
    mining = worker.Worker(dataPath="movies_dataset/cropped.csv", minSupport=min_support, minConfidence=min_confidence)
    start_time = time.time()
    try:
        freq_itemsets, rules, support_dict = mining.runApioriAlgorithmWithRules()
        # Transform freq_itemsets to include support values
        transformed_itemsets = {}
        for k, itemsets in freq_itemsets.items():
            transformed_itemsets[k] = [
                {
                    'items': itemset,
                    'support': support_dict[itemset]
                }
                for itemset in itemsets
            ]
        
        return {
            "frequent_itemsets": transformed_itemsets,
            "rules": rules,
            "time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/apriori_hashtree/custom")
async def run_apriori_hashtree_custom(dataset: CustomDataset):
    start_time = time.time()
    try:
        transactions = dataset.data
        minsup = dataset.min_support
        minconf = dataset.min_confidence
        freq_itemsets, rules = apriori_hash_tree(transactions, minsup, minconf)
        # Chuyển đổi kết quả cho frontend
        transformed_itemsets = {}
        for itemset, support in freq_itemsets.items():
            size = len(itemset)
            if size not in transformed_itemsets:
                transformed_itemsets[size] = []
            transformed_itemsets[size].append({
                "items": list(itemset),
                "support": support
            })
        return {
            "frequent_itemsets": transformed_itemsets,
            "rules": rules,
            "time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/apriori_hashtree/default")
async def run_apriori_hashtree_default(
    min_support: Optional[float] = 0.2,
    min_confidence: Optional[float] = 0.5
):
    import pandas as pd
    start_time = time.time()
    try:
        ratings = pd.read_csv("movies_dataset/cropped.csv")
        transactions = ratings.groupby("userId")["movieId"].apply(list).tolist()
        freq_itemsets, rules = apriori_hash_tree(transactions, min_support, min_confidence)
        transformed_itemsets = {}
        for itemset, support in freq_itemsets.items():
            size = len(itemset)
            if size not in transformed_itemsets:
                transformed_itemsets[size] = []
            transformed_itemsets[size].append({
                "items": list(itemset),
                "support": support
            })
        return {
            "frequent_itemsets": transformed_itemsets,
            "rules": rules,
            "time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/fpgrowth/default")
async def run_fpgrowth_default(
    min_support: Optional[float] = 0.1,
    min_confidence: Optional[float] = 0.3
):
    
    mining = worker.Worker(dataPath="movies_dataset/cropped.csv", minSupport=min_support, minConfidence=min_confidence)
    start_time = time.time()
    try:
        freq_itemsets = mining.runFPGrowthAlgorithm()
        # rules = generateAssociationRules(
        #     frequentItemSets=freq_itemsets,
        #     support_dict=support_dict,
        #     minConfidence=min_confidence
        # )
        return {
            "frequent_itemsets": freq_itemsets,
            # "rules": rules,
            "time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/fpgrowth/custom")
async def run_fpgrowth_custom(dataset: CustomDataset):
    start_time = time.time()
    try:
        transactions = dataset.data
        min_support = dataset.min_support
        min_confidence = dataset.min_confidence
        result = fp_growth(transactions, min_support)
        rules = generate_association_rules_from_fp(result, min_confidence)
        freq_itemsets = {}
        for itemset, support in result.items():
            size = len(itemset)
            if size not in freq_itemsets:
                freq_itemsets[size] = []
            freq_itemsets[size].append({
                "items": list(itemset),
                "support": support
            })
        return {
            "frequent_itemsets": freq_itemsets,
            "rules": rules,
            "time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

ratingTrans = r"C:\Users\Admin\Documents\Last semester\Data Mining Exercise\movies_dataset\cropped.csv"
movies = r"C:\Users\Admin\Documents\Last semester\Data Mining Exercise\processed_movie\movies.csv"
@app.get("/users/unique")
async def get_unique_users():
    try:
        def process_unique_users():
            ratings_df = get_ratings_df()
            return ratings_df['userId'].unique()[:200].tolist()
        
        unique_users = await app.state.loop.run_in_executor(thread_pool, process_unique_users)
        return {
            "unique_users": unique_users,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/{user_id}/ratings")
async def get_user_ratings(user_id: int):
    try:
        def process_user_ratings():
            ratings_df = get_ratings_df()
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            
            if user_ratings.empty:
                raise HTTPException(status_code=404, detail=f"No ratings found for user {user_id}")
            
            ratings_list = []
            for _, row in user_ratings.iterrows():
                rating_dict = {
                    "userId": int(row['userId']),
                    "movieId": int(row['movieId']),
                    "rating": float(row['rating']),
                    "timestamp": int(row['timestamp'])
                }
                ratings_list.append(rating_dict)
            return ratings_list
        
        ratings_list = await app.state.loop.run_in_executor(thread_pool, process_user_ratings)
        return {
            "ratings": ratings_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/movies/{movie_id}")
async def get_movie_metadata(movie_id: str):
    try:
        def process_movie_metadata():
            movies_df = get_movies_df()
            movie_data = movies_df[movies_df["id"].isin([int(id) for id in movie_id.split(',')])]
            if movie_data.empty:
                raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
            
            movie_info = {
                "title": movie_data.iloc[0]['title'],
                "poster_path": movie_data.iloc[0]['poster_path'],
                "overview": movie_data.iloc[0]['overview']
            }
            return movie_info
        
        movie_info = await app.state.loop.run_in_executor(thread_pool, process_movie_metadata)
        return {
            "movie": movie_info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/user/{userId}/recommendations")
async def get_movie_recommend(userId: int):
    try:
        def process_recommendations():
            ratings_df = get_ratings_df()
            movies_df = get_movies_df()
            
            print("Initializing HybridRecommender with n_neighbors=5...")
            cf = HybridRecommender(n_neighbors=5, movies_df=movies_df, ratings_df=ratings_df)
            print("Fitting the model with ratings data...")
            cf.fit(ratings_df, fill_method='mean')
            print("Model fitting completed.")
            print(f"Getting recommendations for user {userId}...")
            recommendations = cf.recommend_items(userId, n_recommendations=5, method='user')
            
            # Convert single recommendation to list if necessary
            if not isinstance(recommendations, list):
                recommendations = [recommendations]
            return recommendations
        
        movie_list = []
        recommendations = await app.state.loop.run_in_executor(thread_pool, process_recommendations)
        
        # Handle both single recommendations and lists of recommendations
        if isinstance(recommendations, list):
            for rec in recommendations:
                if isinstance(rec, tuple):
                    movie_id, rating = rec
                    movie_list.append({
                        "movieId": int(movie_id.item()),
                        "rating": float(rating.item())
                    })
                else:
                    # Handle single movie ID without rating
                    movie_list.append({
                        "movieId": int(rec.item()),
                        "rating": None
                    })
        else:
            # Handle single movie ID without rating
            movie_list.append({
                "movieId": int(recommendations.item()),
                "rating": None
            })
            
        return {
            "recommendations": movie_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.on_event("startup")
async def startup_event():
    app.state.loop = asyncio.get_event_loop()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)