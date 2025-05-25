import numpy as np

def load_similarity_matrix():
    """
    Load and print the similarity matrix from similarity_matrix.npy
    """
    try:
        # Load the similarity matrix
        similarity_matrix = np.load(r'C:\Users\Admin\Documents\Last semester\Data Mining Exercise\utils\hybrid_recommend\similarity_matrix.npy')
        
        # Print the shape and content of the matrix
        print("Matrix shape:", similarity_matrix.shape)
        print("\nMatrix content:")
        print(similarity_matrix)
        
        return similarity_matrix
    except FileNotFoundError:
        print("Error: similarity_matrix.npy file not found")
        return None
    except Exception as e:
        print(f"Error loading matrix: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the function
    load_similarity_matrix()
