import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from content_base import ContentBasedRecommender
from tqdm import tqdm



class HybridRecommender:
    def __init__(self, n_neighbors=5, movies_df=None, ratings_df=None):
        self.n_neighbors = n_neighbors
        self.user_similarity = None
        self.item_similarity = None
        self.ratings_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.mean_ratings = None
        
    def _create_mappings(self, users, items):
        """Create mappings between original IDs and matrix indices"""
        self.user_mapping = {user: idx for idx, user in enumerate(users)}
        self.item_mapping = {item: idx for idx, item in enumerate(items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
    def _findMeanSubtraction(self, ratings_matrix):
        """
        Find the mean subtraction of the ratings matrix, 
        1. Normalize the ratings matrix by subtracting the mean rating for each user from the ratings matrix.
        2. Fill NaN values with the mean rating for each item.
        3. Return the normalized and filled ratings matrix.
        """
        # Calculate user means and normalize
        self.mean_ratings = np.nanmean(ratings_matrix, axis=1)
        normalized_ratings = ratings_matrix - self.mean_ratings[:, np.newaxis]
        
        # Calculate item means and fill NaN values
        item_means = np.nanmean(normalized_ratings, axis=0)
        filled_matrix = np.where(np.isnan(normalized_ratings), item_means, normalized_ratings)
        
        return filled_matrix
    
    def fit(self, ratings_df):
        """
        Fit the collaborative filtering model
        """
        # Create mappings
        unique_users = ratings_df['userId'].unique()
        unique_items = ratings_df['movieId'].unique()
        self._create_mappings(unique_users, unique_items)
        self.ratings_matrix = np.full((len(unique_users), len(unique_items)), np.nan)
        for _, row in ratings_df.iterrows():
            userIdx = self.user_mapping[row['userId']]
            movieIdx = self.item_mapping[row['movieId']]
            self.ratings_matrix[userIdx, movieIdx] = row['rating']
        print(self.ratings_matrix.shape)
        try:
            # Try to load the similarity matrix from file
            self.user_similarity = np.load(r'utils/hybrid_recommend/cf_similarity.npy')
            print("Successfully loaded similarity matrix from file")
        except (FileNotFoundError, IOError):
            print("Similarity matrix file not found, calculating new similarity matrix...")
            # Fill missing values and calculate similarity
            # Create ratings matrix
            filled_matrix = self._findMeanSubtraction(self.ratings_matrix)
            self.user_similarity = self.custom_cosine_similarity(filled_matrix)
            # Save the similarity matrix to file
            np.save(r'utils/hybrid_recommend/cf_similarity.npy', self.user_similarity)
            print("Saved new similarity matrix to file")

    def custom_cosine_similarity(self, user_item_matrix):
        """
        Calculate the cosine similarity of the matrix
        """
        result = np.full((user_item_matrix.shape[0], user_item_matrix.shape[0]), np.nan)
        for i in range(user_item_matrix.shape[0]):
            for j in range(user_item_matrix.shape[0]):
                if j < i:
                    result[i][j] = result[j][i]
                elif i == j:
                    result[i][j] = 1
                else:
                    result[i][j] = self.nan_cosine_similarity(user_item_matrix[i], user_item_matrix[j])
                    
        return result
    def nan_cosine_similarity(self, matrixA, matrixB):
        """
        Calculate the cosine similarity of the matrix
        """
        numerator = np.sum(matrixA * matrixB)
        denominator = np.sqrt(np.sum(matrixA**2) * np.sum(matrixB**2))
        return numerator / denominator
        
        
    def predict_user_based(self, userId, movieId):
        if userId not in self.user_mapping or movieId not in self.item_mapping:
            return None
            
        userIdx = self.user_mapping[userId]
        movieIdx = self.item_mapping[movieId]
        
        # Get similar users
        user_similarities = self.user_similarity[userIdx]
        similar_users = np.argsort(user_similarities)[::-1][1:self.n_neighbors+1]
        
        # Calculate weighted average of ratings
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_userIdx in similar_users:
            if not np.isnan(self.ratings_matrix[similar_userIdx, movieIdx]):
                similarity = user_similarities[similar_userIdx]
                rating = self.ratings_matrix[similar_userIdx, movieIdx]
                weighted_sum += similarity * rating
                similarity_sum += similarity
                
        if similarity_sum == 0:
            return None
            
        return weighted_sum / similarity_sum

    
    def recommend_items(self, userId, n_recommendations=5, method='user'):
        if userId not in self.user_mapping:
            return []

        userIdx = self.user_mapping[userId]
        user_ratings = self.ratings_matrix[userIdx]
        unrated_items = np.where(np.isnan(user_ratings))[0]
        
        if len(unrated_items) == 0:
            return []
        # Initialize and fit the recommender
        cb = ContentBasedRecommender(self.movies_df, self.ratings_df)
        alpha = 0.3
        predictions = []
        for movieIdx in unrated_items:
            movieId = self.reverse_item_mapping[movieIdx]
            if method == 'user':
                pred = self.predict_user_based(userId, movieId)
            if pred is not None:
                print(f"Predicted rating for {userId} and {movieId}: {pred}")
                cb_pred = cb.predict_rating(userId, movieId)

                if cb_pred is not None:
                    print(f"Content-based predicted rating for {movieId}: {cb_pred}")
                    pred = alpha * pred + (1 - alpha) * cb_pred
                    predictions.append((movieId, pred))
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        print(predictions)
        return [movieId for movieId, _ in predictions[:n_recommendations]]
    
        
# Example usage:
if __name__ == "__main__":
    
    # Load data
    ratings_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\movies_dataset\cropped.csv')
    movies_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\processed_movie\movies.csv')
    
    # Initialize and fit the model
    print("Initializing HybridRecommender with n_neighbors=5...")
    cf = HybridRecommender(n_neighbors=5, movies_df=movies_df, ratings_df=ratings_df)
    print("Fitting the model with ratings data...")
    cf.fit(ratings_df)
    print("Model fitting completed.")
    
    # cf._plot_similarity_matrices()
    # Example: Get recommendations for a user
    userId = 46
    print(f"Getting recommendations for user {userId}...")
    recommendations = cf.recommend_items(userId, n_recommendations=5, method='user')
    print("Recommendations generated:")
    print(recommendations)
    # Print recommendations
    print(f"Top 5 recommendations for user {userId}:")
    for movieId in recommendations:
        movie_title = movies_df[movies_df['id'] == movieId]['title'].values[0]
        print(f"- {movie_title}") 