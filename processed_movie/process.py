import pandas as pd

# Read the data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Convert movie IDs to numeric, handling any errors
movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
ratings_df['movieId'] = pd.to_numeric(ratings_df['movieId'], errors='coerce')

# Get the intersection of movie IDs from both dataframes
common_movie_ids = set(movies_df['id']).intersection(set(ratings_df['movieId']))

# Filter both dataframes to keep only records with common movie IDs
movies_df = movies_df[movies_df['id'].isin(common_movie_ids)]
ratings_df = ratings_df[ratings_df['movieId'].isin(common_movie_ids)]

# Save the processed data
movies_df.to_csv('processed_movies.csv', index=False)
ratings_df.to_csv('processed_ratings.csv', index=False)

# Save first 10000 records of ratings to a new file
ratings_df.head(10000).to_csv('rating_smalls.csv', index=False)

# Check problematic movie IDs
# problematic_ids = [349, 380, 588, 595, 899]
# print("\nChecking problematic movie IDs:")
# for movie_id in problematic_ids:
#     movie_exists = movie_id in movies_df['id'].values
#     print(f"Movie ID {movie_id} exists in movies_df: {movie_exists}")
#     if movie_exists:
#         print(f"Title: {movies_df[movies_df['id'] == movie_id]['title'].values[0]}")

print(f"\nProcessed movies shape: {movies_df.shape}")
print(f"Processed ratings shape: {ratings_df.shape}")
print(f"Small ratings file shape: {ratings_df.head(10000).shape}")
