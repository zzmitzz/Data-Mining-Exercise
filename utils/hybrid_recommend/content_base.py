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
        self.full_matrix = None
        self.genre_similarity_matrix = None
        self.keyword_similarity_matrix = None
        self.genre_list = None  # Store list of all genres
        
        # Create movie ID to index mappings
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self._create_movie_mappings()
        
    def _create_movie_mappings(self):
        """
        Create mappings between movie IDs and their indices in the dataframe
        """
        for idx, movie_id in enumerate(self.movies_df['id']):
            self.movie_id_to_idx[movie_id] = idx
            self.idx_to_movie_id[idx] = movie_id
            
    def get_movie_idx(self, movie_id):
        """
        Get the index of a movie given its ID
        
        Args:
            movie_id: The ID of the movie
            
        Returns:
            int: The index of the movie in the dataframe
        """
        return self.movie_id_to_idx.get(movie_id)
        
    def get_movie_id(self, idx):
        """
        Get the movie ID given its index
        
        Args:
            idx: The index of the movie in the dataframe
            
        Returns:
            int: The ID of the movie
        """
        return self.idx_to_movie_id.get(idx)

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
            
    def fit(self):
            
        print("Creating sentence embeddings for movie overviews...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(self.movies_df["overview"].fillna("").tolist(), show_progress_bar=True)
        
        print("Computing and saving similarity matrix...")
        self.similarity_matrix = cosine_similarity(embeddings)
        
        # Save in both binary and text format
        np.save('overview_similarity_matrix.npy', self.similarity_matrix)
        
        return self.similarity_matrix

    def predict_rating(self, user_id, movie_id, full_matrix):
        # Get user's previous ratings
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
            
        # Get movie index using the mapping
        movie_idx = self.get_movie_idx(movie_id)
        if movie_idx is None:
            print(f"Movie {movie_id} not found in movies_df")
            return None
        
        # Read only the specific row from similarity matrix
        try:
            # Get the specific row
            similarity_scores = full_matrix[movie_idx]
            
        except FileNotFoundError:
            raise ValueError("Similarity matrix not found. Please call fit() first.")
        except Exception as e:
            print(f"Error reading similarity matrix: {str(e)}")
            raise
        
        # Calculate weighted average of ratings
        weighted_sum = 0
        similarity_sum = 0
        
        print(f"Computing prediction for user {user_id} and movie {movie_id}...")
        for _, rating in tqdm(user_ratings.iterrows(), total=len(user_ratings), desc="Processing user ratings"):
            rated_movie_idx = self.get_movie_idx(rating['movieId'])
            if rated_movie_idx is not None:
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

    def fit_keyword_based(self, movies_df=None):
        if movies_df is not None:
            self.movies_df = movies_df
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        # Fill NaN values with empty strings and convert to list
        list_keywords = self.movies_df['keywords'].fillna("").tolist()
        
        # Fit the vectorizer on the keywords
        keywords_matrix = vectorizer.fit_transform(list_keywords)
        
        # Compute the cosine similarity matrix
        self.keyword_similarity_matrix = cosine_similarity(keywords_matrix)
        
        # Save the similarity matrix
        np.save('keyword_similarity_matrix.npy', self.keyword_similarity_matrix)
        
        return self.keyword_similarity_matrix

    def combine_similarity_matrices(self, matrix1_path, matrix2_path, output_path='similar_item_item_matrix.npy'):
        """
        Load two similarity matrices and combine them into one item-item matrix.
        
        Args:
            matrix1_path (str): Path to the first similarity matrix (.npy file)
            matrix2_path (str): Path to the second similarity matrix (.npy file)
            output_path (str): Path to save the combined matrix
            
        Returns:
            numpy.ndarray: The combined similarity matrix
        """
        try:
            # Load both matrices
            print("Loading first similarity matrix...")
            matrix1 = np.load(matrix1_path)
            print("Loading second similarity matrix...")
            matrix2 = np.load(matrix2_path)
            
            # Ensure both matrices have the same shape
            if matrix1.shape != matrix2.shape:
                raise ValueError("Matrices must have the same shape")
            
            # Combine matrices (using simple average)
            print("Combining matrices...")
            combined_matrix = (matrix1 + matrix2) / 2
            
            # Save the combined matrix
            print(f"Saving combined matrix to {output_path}...")
            np.save(output_path, combined_matrix)
            np.savetxt("combined_matrix.txt", combined_matrix, fmt='%.2f')
            return combined_matrix
            
        except FileNotFoundError as e:
            print(f"Error: Could not find one of the similarity matrices: {str(e)}")
            return None
        except Exception as e:
            print(f"Error combining matrices: {str(e)}")
            return None

    def recommend_items(self, user_id, similarity_matrix=None):
        """
        Generate movie recommendations for a user based on their rated movies and similarity scores.
        
        Args:
            user_id: The ID of the user to get recommendations for
            similarity_matrix: The similarity matrix to use (if None, uses self.similarity_matrix)
            
        Returns:
            List of tuples (movieId, predicted_rating) for the top N recommendations
        """
        if similarity_matrix is None:
            similarity_matrix = self.similarity_matrix
            
        if similarity_matrix is None:
            raise ValueError("No similarity matrix available. Please load or compute one first.")
            
        # Get user's previous ratings
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            print(f"No ratings found for user {user_id}")
            return []
            
        # Calculate mean rating for this user
        mean_rating = user_ratings['rating'].mean()
        
        # Get all movies the user hasn't rated
        all_movie_ids = set(self.movies_df['id'].astype(int))
        rated_movie_ids = set(user_ratings['movieId'].astype(int))
        unrated_movie_ids = all_movie_ids - rated_movie_ids
        print(unrated_movie_ids)
        predictions = []
        print(f"Generating recommendations for user {user_id}...")
        
        # For each unrated movie
        for movie_id in tqdm(unrated_movie_ids, desc="Processing movies"):
            movie_idx = self.get_movie_idx(int(movie_id))
            if movie_idx is None:
                continue
                
            # Get similarity scores for this movie
            similarity_scores = similarity_matrix[movie_idx]
            
            # Calculate weighted score based on user's ratings
            weighted_sum = 0
            similarity_sum = 0
            
            for _, rating in user_ratings.iterrows():
                rated_movie_idx = self.get_movie_idx(int(rating['movieId']))
                if rated_movie_idx is not None:
                    similarity = similarity_scores[rated_movie_idx]
                    # Subtract mean from rating before weighting
                    normalized_rating = rating['rating'] - mean_rating
                    weighted_sum += similarity * normalized_rating
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                # Calculate prediction and add back the mean
                predicted_rating = (weighted_sum / similarity_sum) + mean_rating
                # Ensure rating is within bounds
                predicted_rating = min(max(predicted_rating, 0), 5)
                predictions.append((int(movie_id), predicted_rating))
            else:
                predictions.append((int(movie_id), mean_rating))  # Use mean rating as fallback
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions, predictions[:5]

if __name__ == "__main__":
    ratings_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\processed_movie\rating_smalls.csv')
    movies_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\movies_dataset\movies_with_keywords.csv')
    # Initialize and fit the recommender
    print("Initializing recommender...")
    cb = ContentBasedRecommender(movies_df, ratings_df)
    

    # # Find the similarity matrix
    # cb.fit()
    # cb.fit_keyword_based(movies_df)
    # cb.combine_similarity_matrices(
    #     r"C:\Users\Admin\Documents\Last semester\Data Mining Exercise\utils\hybrid_recommend\weight\keyword_similarity_matrix.npy",
    #     r"C:\Users\Admin\Documents\Last semester\Data Mining Exercise\utils\hybrid_recommend\weight\overview_similarity_matrix.npy"
    # )

    user_id = 5
    similarity_matrix = np.load(r"C:\Users\Admin\Documents\Last semester\Data Mining Exercise\utils\hybrid_recommend\weight\similar_item_item_matrix.npy")
    recommendations, top_recommendations = cb.recommend_items(user_id, similarity_matrix=similarity_matrix)
    # Print recommendations with movie titles
    # print()
    print(f"\nTop 5 recommendations for user {user_id}:")
    for movieId, pred in top_recommendations:
        try:
            # Convert movieId to integer if it's not already
            movieId = int(movieId)
            # Convert to string for comparison since movies_df['id'] might be string
            movie_title = movies_df[movies_df['id'] == movieId]['title'].values[0]
            print(f"- {movie_title}  {movieId}  with predicted rating {pred:.2f}")
        except (IndexError, KeyError, ValueError):
            print(f"- Movie ID {movieId} with predicted rating {pred:.2f} (title not found)") 

