import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



class ContentBasedRecommender:
    def __init__(self, movies_df, ratings_df=None):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.similarity_matrix = None
        self.tfidf_matrix = None
        
    def preprocess_text(self, text):
        """
        Preprocess text overview
        """
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            # Remove special characters
            text = re.sub(r'[^\w\s]', '', text)
            # Remove stopwords
            try:
                stop_words = set(stopwords.words('english'))
                text = ' '.join([word for word in text.split() if word not in stop_words])
            except:
                nltk.download('stopwords')
                stop_words = set(stopwords.words('english'))
                text = ' '.join([word for word in text.split() if word not in stop_words])
            return text
        else:
            return ""
            
    def fit(self, ratings_df=None):
        """
        Fit the recommender system with ratings data and compute similarity matrix
        """
        if ratings_df is not None:
            self.ratings_df = ratings_df
            
        print("Processing movie overviews...")
        self.movies_df["processed_overview"] = self.movies_df["overview"].apply(self.preprocess_text)
        
        print("Creating sentence embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.movies_df["processed_overview"] = self.movies_df["processed_overview"].apply(model.encode)
        print(self.movies_df["processed_overview"])
        print("Computing similarity matrix...")
    
        self.similarity_matrix = model.similarity(self.movies_df["processed_overview"], self.movies_df["processed_overview"])
        print("Saving similarity matrix...")
        np.save('similarity_matrix.npy', self.similarity_matrix)
        
        return self.similarity_matrix

    def predict_rating(self, user_id, movie_id):
        """
        Predict a rating for a given user and movie based on their previous ratings
        
        Args:
            user_id: The ID of the user
            movie_id: The ID of the movie to predict rating for
            
        Returns:
            float: Predicted rating for the movie
        """
        # Get user's previous ratings
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
            
        # Get movie index
        movie_idx = self.movies_df[self.movies_df['id'] == movie_id].index
        if len(movie_idx) == 0:
            print(f"Movie {movie_id} not found in movies_df")
            return None
            
        movie_idx = movie_idx[0]
        
        # Read only the specific row from similarity matrix
        try:
            matrix_shape = (45466, 45466)  # Fixed shape of the similarity matrix
            with open(r"C:\Users\Admin\Documents\Last semester\Data Mining Exercise\utils\hybrid_recommend\similarity_matrix.npy", 'rb') as f:
                # Skip NPY header (128 bytes)
                f.seek(128)
                # Calculate offset to the specific row
                row_offset = movie_idx * matrix_shape[1] * 8  # 8 bytes per float64
                f.seek(128 + row_offset)
                # Read exactly one row (45466 elements)
                similarity_scores = np.fromfile(f, dtype='float64', count=matrix_shape[1])
                print("Similarity scores: ", similarity_scores)
        except FileNotFoundError:
            raise ValueError("Similarity matrix not found. Please call fit() first.")
        
        # Calculate weighted average of ratings
        weighted_sum = 0
        similarity_sum = 0
        
        print(f"Computing prediction for user {user_id} and movie {movie_id}...")
        for _, rating in tqdm(user_ratings.iterrows(), total=len(user_ratings), desc="Processing user ratings"):
            rated_movie_idx = self.movies_df[self.movies_df['id'] == rating['movieId']].index
            if len(rated_movie_idx) > 0:
                rated_movie_idx = rated_movie_idx[0]
                similarity = similarity_scores[rated_movie_idx]
                weighted_sum += similarity * rating['rating']
                similarity_sum += similarity
                
        if similarity_sum == 0:
            return 0
            
        return weighted_sum / similarity_sum
    
    def load_similarity_matrix(self, filepath='similarity_matrix.npy', ratings_df=None):
        if ratings_df is not None:
            self.ratings_df = ratings_df
        try:
            print("Loading similarity matrix...")
            self.similarity_matrix = np.load(filepath)
            return self.similarity_matrix
        except FileNotFoundError:
            print(f"Error: Could not find similarity matrix at {filepath}")
            return None
    
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    movies_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\processed_movie\movies.csv')
    ratings_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\movies_dataset\cropped.csv')
    
    # Initialize and fit the recommender
    print("Initializing recommender...")
    cb = ContentBasedRecommender(movies_df, ratings_df)
    user_id = 1
    movie_id = 2018
    cb.fit(ratings_df)
    predicted_rating = cb.predict_rating(user_id, movie_id)
    print(f"Predicted rating for user {user_id} and movie {movie_id}: {predicted_rating}")
