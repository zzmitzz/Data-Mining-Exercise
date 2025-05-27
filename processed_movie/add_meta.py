import pandas as pd
import ast

def map_movie_keywords():
    # Read the movies.csv and keywords.csv files
    movies_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\processed_movie\processed_movies.csv')
    keywords_df = pd.read_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\movies_dataset\keywords.csv')
    
    # Create a dictionary to store movie keywords
    movie_keywords = {}
    
    # Process each row in the keywords dataframe
    for _, row in keywords_df.iterrows():
        movie_id = row['id']
        keywords = row['keywords']
        
        # Convert string representation of list to actual list if needed
        if isinstance(keywords, str):
            try:
                keywords_list = ast.literal_eval(keywords)
                # Extract just the keyword names and join them with spaces
                keyword_names = [k['name'] for k in keywords_list]
                movie_keywords[movie_id] = ' '.join(keyword_names)
            except:
                movie_keywords[movie_id] = keywords
        else:
            movie_keywords[movie_id] = str(keywords)
    
    # Add the concatenated keywords as a new column
    movies_df['keywords'] = movies_df['id'].map(movie_keywords)
    
    # Save the updated dataframe back to CSV
    movies_df.to_csv(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\processed_movie\movies_with_keywords.csv', index=False)
    
    return movie_keywords

if __name__ == "__main__":
    keyword_map = map_movie_keywords()
    print("Keywords mapping completed successfully!")

