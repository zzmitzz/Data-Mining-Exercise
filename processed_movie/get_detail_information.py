import requests
from bs4 import BeautifulSoup
import json
import time

def get_movie_details(imdb_id):
    """
    Get movie details from IMDB using the movie ID
    Args:
        imdb_id (str): IMDB ID of the movie (e.g., 'tt0111161')
    Returns:
        dict: Dictionary containing movie details
    """
    # Construct the URL
    url = f"https://www.imdb.com/title/{imdb_id}/"
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract movie details
        movie_data = {
            'title': soup.find('h1').text.strip() if soup.find('h1') else None,
            'tagline': soup.find('div', {'data-testid': 'plot-xl'}).text.strip() if soup.find('div', {'data-testid': 'plot-xl'}) else None,
            'genres': [genre.text.strip() for genre in soup.find_all('a', {'class': 'ipc-chip ipc-chip--on-baseAlt'})] if soup.find_all('a', {'class': 'ipc-chip ipc-chip--on-baseAlt'}) else [],
            'keywords': [keyword.text.strip() for keyword in soup.find_all('a', {'class': 'ipc-chip ipc-chip--on-baseAlt'})] if soup.find_all('a', {'class': 'ipc-chip ipc-chip--on-baseAlt'}) else []
        }
        
        return movie_data
    
    except requests.RequestException as e:
        print(f"Error fetching data for IMDB ID {imdb_id}: {str(e)}")
        return None

def main():
    # Example usage
    imdb_id = "tt0111161"  # Example: The Shawshank Redemption
    movie_details = get_movie_details(imdb_id)
    
    if movie_details:
        print(json.dumps(movie_details, indent=4))
    else:
        print("Failed to fetch movie details")

if __name__ == "__main__":
    main()
