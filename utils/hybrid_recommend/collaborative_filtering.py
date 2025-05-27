import numpy as np
import pandas as pd



class CollaborativeFiltering:
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
        Find the mean subtraction of the ratings matrix using KNN approach:
        1. Normalize the ratings matrix by subtracting the mean rating for each user
        2. For each missing value, find the 5 most similar users who rated that item
        3. Fill missing values using weighted average of similar users' ratings
        """
        # Calculate user means and normalize
        self.mean_ratings = np.nanmean(ratings_matrix, axis=1)
        normalized_ratings = ratings_matrix - self.mean_ratings[:, np.newaxis]
        
        # Calculate user similarity matrix
        user_similarity = self.custom_cosine_similarity(normalized_ratings)
        
        # Fill missing values using KNN
        filled_matrix = normalized_ratings.copy()
        n_users = normalized_ratings.shape[0]
        n_items = normalized_ratings.shape[1]
        
        for user_idx in range(n_users):
            for item_idx in range(n_items):
                if np.isnan(normalized_ratings[user_idx, item_idx]):
                    # Get similar users who rated this item
                    similar_users = []
                    for other_user in range(n_users):
                        if not np.isnan(normalized_ratings[other_user, item_idx]):
                            similarity = user_similarity[user_idx, other_user]
                            rating = normalized_ratings[other_user, item_idx] + self.mean_ratings[other_user]
                            similar_users.append((similarity, rating))
                    
                    # Sort by similarity and take top 5
                    similar_users.sort(reverse=True)
                    top_k = similar_users[:5]
                    
                    if top_k:
                        # Calculate weighted average
                        weighted_sum = sum(sim * rating for sim, rating in top_k)
                        similarity_sum = sum(sim for sim, _ in top_k)
                        filled_matrix[user_idx, item_idx] = weighted_sum / similarity_sum if similarity_sum > 0 else 0
                    else:
                        # If no similar users found, use item mean
                        item_mean = np.nanmean(normalized_ratings[:, item_idx])
                        filled_matrix[user_idx, item_idx] = item_mean if not np.isnan(item_mean) else 0
        
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
            self.user_similarity = np.load(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\utils\hybrid_recommend\weight\cf-user-user-similarity.npy')
            print("Successfully loaded similarity matrix from file")
        except (FileNotFoundError, IOError):
            print("Similarity matrix file not found, calculating new similarity matrix...")
            filled_matrix = self._findMeanSubtraction(self.ratings_matrix)
            self.user_similarity = self.custom_cosine_similarity(filled_matrix, nan_value=True)
            # Save the similarity matrix to file
            np.save(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\utils\hybrid_recommend\weight\cf-user-user-similarity.npy', self.user_similarity)
            # Also save as text file for easier viewing
            print("Saved new similarity matrix to file")

    def custom_cosine_similarity(self, user_item_matrix, nan_value=False):
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
                    result[i][j] = self.nan_cosine_similarity(user_item_matrix[i], user_item_matrix[j], nan_value)
                    
        return result
    def nan_cosine_similarity(self, matrixA, matrixB, nan_value=False):
        """
        Calculate the cosine similarity between two vectors, handling NaN values.
        Only considers ratings where both users have rated the item (no NaN in either vector).
        """
        # Create mask for non-NaN values in both vectors
        mask = ~(np.isnan(matrixA) | np.isnan(matrixB))
        
        # If no common ratings, return 0 similarity
        if not np.any(mask):
            if nan_value:
                return np.nan
            else:
                return 0
            
        # Get only the common ratings
        a = matrixA[mask]
        b = matrixB[mask]
        
        # Calculate cosine similarity only on common ratings
        numerator = np.sum(a * b)
        denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
        
        # Handle case where denominator is 0
        if denominator == 0:
            if nan_value:
                return np.nan
            else:
                return 0
            
        return numerator / denominator
        
        
    def predict_user_based(self, userId, movieId):
        if userId not in self.user_mapping or movieId not in self.item_mapping:
            return None
            
        userIdx = self.user_mapping[userId]
        movieIdx = self.item_mapping[movieId]
        
        # Get similar users
        user_similarities = self.user_similarity[userIdx]
        similar_users = np.argsort(user_similarities)[::-1]
        if(len(similar_users) > 50):
            similar_users = similar_users[:50]
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

    
    def recommend_items(self, userId, n_recommendations=5):
        """
        Recommend items for a user based on collaborative filtering
        Args:
            userId: The ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
        Returns:
            List of tuples (movieId, predicted_rating) for the top N recommendations
        """
        
        print("Fitting the model with ratings data...")
        self.fit(self.ratings_df)
        if userId not in self.user_mapping:
            return []
        
        userIdx = self.user_mapping[userId]
        user_ratings = self.ratings_matrix[userIdx]
        unrated_items = np.where(np.isnan(user_ratings))[0]
        
        if len(unrated_items) == 0:
            return []

        predictions = []
        for movieIdx in unrated_items:
            # Convert index back to original movieId
            movieId = self.reverse_item_mapping[movieIdx]
            pred = self.predict_user_based(userId, movieId)
            # If prediction is None, assign minimum integer value
            if pred is None or np.isnan(pred):
                pred = 0
            predictions.append((movieId, pred))

        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        # np.savetxt(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\utils\hybrid_recommend\weight\predictions.txt', predictions, fmt='%d %.2f')
        return predictions, predictions[:n_recommendations]

# Example usage:
if __name__ == "__main__":
    
    # Load data
    ratings_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\movies_dataset\ratings_small.csv')
    movies_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\movies_dataset\movies_with_keywords.csv')
    
    # Initialize and fit the model
    print("Initializing HybridRecommender with n_neighbors=5...")
    cf = CollaborativeFiltering(n_neighbors=5, movies_df=movies_df, ratings_df=ratings_df)
    
    
    # Example: Get recommendations for a user
    userId = 5
    print(f"Getting recommendations for user {userId}...")
    recommendations, top_recommendations = cf.recommend_items(userId, n_recommendations=5)
    # Print recommendations with movie titles
    print(f"\nTop 5 recommendations for user {userId}:")
    for movieId, pred in top_recommendations:
        try:
            # Convert movieId to integer if it's not already
            movieId = int(movieId)
            # Convert to string for comparison since movies_df['id'] might be string
            movie_title = movies_df[movies_df['id'].astype(str) == str(movieId)]['title'].values[0]
            print(f"- {movie_title} with predicted rating {pred:.2f}")
        except (IndexError, KeyError, ValueError):
            print(f"- Movie ID {movieId} with predicted rating {pred:.2f} (title not found)") 