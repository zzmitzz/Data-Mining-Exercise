import pandas as pd

# Read the data
movies_df = pd.read_csv('movies_metadata.csv')
ratings_df = pd.read_csv('ratings.csv')

# Convert movie IDs to numeric, handling any errors
movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
ratings_df['movieId'] = pd.to_numeric(ratings_df['movieId'], errors='coerce')

# Filter ratings where movieId <= 2000
# ratings_df = ratings_df[ratings_df['movieId'] <= 10000]

# # Filter movies where id <= 2000
# movies_df = movies_df[movies_df['id'] <= 10000]

# Keep only ratings that have corresponding movie IDs in the metadata
ratings_df = ratings_df[ratings_df['movieId'].isin(movies_df['id'])]

# Save the processed data
movies_df.to_csv('processed_movies.csv', index=False)
ratings_df.to_csv('processed_ratings.csv', index=False)

print(f"Processed movies shape: {movies_df.shape}")
print(f"Processed ratings shape: {ratings_df.shape}")