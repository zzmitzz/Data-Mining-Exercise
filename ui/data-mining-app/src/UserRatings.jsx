import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';

function UserRatings() {
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(1);
  const [userRatings, setUserRatings] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [movieDetails, setMovieDetails] = useState(null);
  const [isLoadingUsers, setIsLoadingUsers] = useState(false);
  const [isLoadingRatings, setIsLoadingRatings] = useState(false);
  const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(false);
  const [isLoadingMovieDetails, setIsLoadingMovieDetails] = useState(false);
  const [movieDetailsCache, setMovieDetailsCache] = useState({});
  const [lastFetchTime, setLastFetchTime] = useState(0);
  const [currentRequest, setCurrentRequest] = useState(null);
  const navigate = useNavigate();


  const errorImageAlternateLink = [
    "https://img.freepik.com/free-vector/professional-suspense-movie-poster_742173-3470.jpg?semt=ais_hybrid&w=740",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIaYBaFEhB0VQUOtQ0LiQbSkg33uZXV07Hpg&s",
    "https://intheposter.com/cdn/shop/files/the-manager-in-the-poster-1.jpg?v=1733910535",
    "https://m.media-amazon.com/images/I/71qu4p5bnDL._AC_UF894,1000_QL80_.jpg",
    "https://d1csarkz8obe9u.cloudfront.net/posterpreviews/action-movie-poster-template-design-0f5fff6262fdefb855e3a9a3f0fdd361_screen.jpg?ts=1700270983",
    "https://artofthemovies.co.uk/cdn/shop/files/IMG_4154_1-780453_de0cc110-550d-4448-a7ec-d3ff945c0739.jpg?v=1696169470",
    "https://i.ytimg.com/vi/lKe2v-uBKBI/maxresdefault.jpg",
    "https://images.photowall.com/products/51078/movie-poster-jaws.jpg?h=699&q=85",
    "https://intheposter.com/cdn/shop/products/the-front-line-in-the-poster-1.jpg?v=1733910578",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmo4IXxBfM3ovhzY2XtwoDUl6IvEAAKS7_5A&s",
    "https://www.tallengestore.com/cdn/shop/products/1917_-_Sam_Mendes_-_Hollywood_War_Film_Classic_English_Movie_Poster_a12704bd-2b25-4aa7-8c8d-8f40cf467dc7.jpg?v=1582781089",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSztWlWx9Cu6OHmJXBGy6kwU3QugErKM09O7g&s"
  ]

  const errorImageLink2 = [
    [
      "https://m.media-amazon.com/images/M/MV5BM2MyNjYxNmYtYTAwZi00ZjQxLWFmNDEtODM1ZTA5YzI0YzM0XkEyXkFqcGdeQXVyNDYyMDk5MTU@._V1_.jpg",
      "https://m.media-amazon.com/images/M/MV5BMTY5ODczODQ2M15BMl5BanBnXkFtZTgwNzY0NTUyMDE@._V1_.jpg",
      "https://m.media-amazon.com/images/M/MV5BNzA5ZDNjYzMtYzZkYy00ODNlLWE4MjUtNjQ1ZjJmMzRlMmZlXkEyXkFqcGdeQXVyMTA1NTM1NDI2._V1_FMjpg_UX1000_.jpg",
      "https://m.media-amazon.com/images/M/MV5BY2U5OTgxMTgtZDA1Ni00NzU2LWIwZTAtZmJkMzJjNTE1NjY2XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_.jpg",
      "https://m.media-amazon.com/images/M/MV5BZjdkOTU3MzMtYzA1ZC00ZDg4LWE4NjgtMDc1NzIwODhkZmU2XkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_FMjpg_UX1000_.jpg"
    ]
    
  ]
  // Fetch users on component mount
  useEffect(() => {
    console.log("Fetching users");
    fetchUsers();
  }, []);

  // Fetch user ratings when a user is selected
  useEffect(() => {
    let isMounted = true;
    
    const fetchData = async () => {
      if (selectedUser && isMounted) {
        await fetchUserRatings(selectedUser);
        await fetchRecommendations(selectedUser);
      }
    };

    fetchData();

    return () => {
      isMounted = false;
    };
  }, [selectedUser]);

  const fetchUsers = async () => {
    setIsLoadingUsers(true);
    try {
      const response = await fetch('http://localhost:8000/users/unique');
      if (!response.ok) throw new Error('Failed to fetch users');
      const data = await response.json();
      setUsers(data["unique_users"]);
      console.log(data["unique_users"]);
    } catch (error) {
      console.error('Error fetching users:', error);
    } finally {
      setIsLoadingUsers(false);
    }
  };

  const fetchUserRatings = async (userId) => {
    setIsLoadingRatings(true);
    try {
      const response = await fetch(`http://localhost:8000/users/${userId}/ratings`);
      if (!response.ok) throw new Error('Failed to fetch user ratings');
      const data = await response.json();
      setUserRatings(data["ratings"]);
    } catch (error) {
      console.error('Error fetching user ratings:', error);
    } finally {
      setIsLoadingRatings(false);
    }
  };

  const fetchRecommendations = async (userId) => {
    setIsLoadingRecommendations(true);
    try {
      const response = await fetch(`http://localhost:8000/user/${userId}/recommendations`);
      if (!response.ok) throw new Error('Failed to fetch recommendations');
      const data = await response.json();
      setRecommendations(data["recommendations"]);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    } finally {
      setIsLoadingRecommendations(false);
    }
  };

  const fetchMovieDetails = useCallback(async (movieId) => {
    // Check cache first
    if (movieDetailsCache[movieId]) {
      setMovieDetails(movieDetailsCache[movieId]);
      return;
    }

    // Cancel any ongoing request
    if (currentRequest) {
      currentRequest.abort();
    }

    // Create new AbortController for this request
    const controller = new AbortController();
    setCurrentRequest(controller);

    // Debounce requests (minimum 500ms between requests)
    const now = Date.now();
    if (now - lastFetchTime < 500) {
      return;
    }
    setLastFetchTime(now);

    setIsLoadingMovieDetails(true);
    try {
      const response = await fetch(`http://localhost:8000/movies/${movieId}`, {
        signal: controller.signal
      });
      if (!response.ok) throw new Error('Failed to fetch movie details');
      const data = await response.json();
      const movieData = data["movie"];
      
      // Update cache
      setMovieDetailsCache(prev => ({
        ...prev,
        [movieId]: movieData
      }));
      
      setMovieDetails(movieData);
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request was cancelled');
        return;
      }
      console.error('Error fetching movie details:', error);
    } finally {
      setIsLoadingMovieDetails(false);
      setCurrentRequest(null);
    }
  }, [movieDetailsCache, lastFetchTime, currentRequest]);

  // Cleanup function to cancel any ongoing request when component unmounts
  useEffect(() => {
    return () => {
      if (currentRequest) {
        currentRequest.abort();
      }
    };
  }, [currentRequest]);

  const handleMovieClick = useCallback((movie) => {
    setSelectedMovie(movie);
    fetchMovieDetails(movie.movieId);
  }, [fetchMovieDetails]);

  const MovieCard = ({ movie, onClick, errorImageAlternateLink }) => {
    const [movieInfo, setMovieInfo] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
      const fetchMovieInfo = async () => {
        // Check cache first
        if (movieDetailsCache[movie.movieId]) {
          setMovieInfo(movieDetailsCache[movie.movieId]);
          setIsLoading(false);
          return;
        }

        try {
          const response = await fetch(`http://localhost:8000/movies/${movie.movieId}`);
          if (!response.ok) throw new Error('Failed to fetch movie details');
          const data = await response.json();
          const movieData = data.movie;
          
          // Update cache
          setMovieDetailsCache(prev => ({
            ...prev,
            [movie.movieId]: movieData
          }));
          
          setMovieInfo(movieData);
        } catch (error) {
          console.error('Error fetching movie details:', error);
        } finally {
          setIsLoading(false);
        }
      };

      fetchMovieInfo();
    }, [movie.movieId, movieDetailsCache]);

    if (isLoading) {
      return (
        <div className="movie-card p-4 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl shadow-lg border border-indigo-100 min-w-[200px] h-[100px] flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-4 border-indigo-200 border-t-indigo-600"></div>
        </div>
      );
    }

    return (
      <div 
        className="movie-card cursor-pointer p-4 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-indigo-100 min-w-[200px]"
        onClick={() => onClick(movie)}
      >
        <div className="text-center">
          <h3 className="text-lg font-bold text-indigo-800 mb-2 line-clamp-2">{movieInfo?.title || 'Unknown Title'}</h3>
          <div className="flex items-center justify-center gap-1">
            {[...Array(5)].map((_, index) => (
              <span key={index} className={`text-xl ${index < Math.round(movie.rating) ? 'text-yellow-400' : 'text-gray-200'}`}>
                ★
              </span>
            ))}
          </div>
          <span className="text-indigo-600 mt-1 block font-medium">({movie.rating.toFixed(1)}/5)</span>
        </div>
      </div>
    );
  };

  const LoadingSpinner = () => (
    <div className="flex justify-center items-center p-8">
      <div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-200 border-t-indigo-600"></div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 p-8">
      <button 
        onClick={() => navigate('/')}
        className="mb-8 px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl flex items-center gap-2 transform hover:-translate-y-0.5"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M9.707 14.707a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 1.414L7.414 9H15a1 1 0 110 2H7.414l2.293 2.293a1 1 0 010 1.414z" clipRule="evenodd" />
        </svg>
        Back to Main
      </button>

      <div className="flex gap-8">
        {/* Left Panel */}
        <div className="w-2/3">
          <div className="mb-8 bg-white/80 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-indigo-100">
            <label className="block text-2xl font-bold mb-4 text-indigo-800">Select User</label>
            {isLoadingUsers ? (
              <LoadingSpinner />
            ) : (
              <select 
                value={selectedUser || ''} 
                onChange={(e) => setSelectedUser(e.target.value)}
                className="w-full p-4 border-2 border-indigo-200 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all duration-300 appearance-none bg-white text-gray-700 font-medium"
              >
                <option value="" className="text-gray-500">Select a user...</option>
                {users.map(user => (
                  <option key={user} value={user} className="text-gray-700">
                    User {user}
                  </option>
                ))}
              </select>
            )}
          </div>

          {selectedUser && (
            <>
              <div className="mb-10">
                <h2 className="text-3xl font-bold mb-6 text-indigo-800 flex items-center gap-2">
                  <span className="text-purple-600">★</span> User's Rated Movies
                </h2>
                {isLoadingRatings ? (
                  <LoadingSpinner />
                ) : (
                  <div className="flex overflow-x-auto gap-6 pb-6 scrollbar-thin scrollbar-thumb-indigo-300 scrollbar-track-transparent hover:scrollbar-thumb-indigo-400">
                    {userRatings.map((movie, index) => (
                      <MovieCard 
                        key={movie.movieId} 
                        movie={movie} 
                        onClick={handleMovieClick}
                        errorImageAlternateLink={errorImageAlternateLink[index % errorImageAlternateLink.length]}
                      />
                    ))}
                  </div>
                )}
              </div>

              <div>
                <h2 className="text-3xl font-bold mb-6 text-indigo-800 flex items-center gap-2">
                  <span className="text-purple-600">🎬</span> Recommended Movies
                </h2>
                {isLoadingRecommendations ? (
                  <LoadingSpinner />
                ) : (
                  <div className="flex overflow-x-auto gap-6 pb-6 scrollbar-thin scrollbar-thumb-indigo-300 scrollbar-track-transparent hover:scrollbar-thumb-indigo-400">
                    {recommendations.map((movie, index) => (
                      <MovieCard 
                        key={movie.movieId} 
                        movie={movie} 
                        onClick={handleMovieClick}
                        errorImageAlternateLink={errorImageLink2[index % errorImageLink2.length]}
                      />
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* Right Panel */}
        <div className="w-1/3">
          {isLoadingMovieDetails ? (
            <div className="bg-white/80 backdrop-blur-sm p-8 rounded-xl shadow-lg sticky top-8 border border-indigo-100">
              <LoadingSpinner />
            </div>
          ) : movieDetails && (
            <div className="bg-white/80 backdrop-blur-sm p-8 rounded-xl shadow-lg sticky top-8 border border-indigo-100">
              <div className="relative overflow-hidden rounded-lg mb-6 group">
                <img 
                  src={`https://image.tmdb.org/t/p/w500${movieDetails.poster_path}`}
                  alt={movieDetails.title} 
                  className="w-full h-[500px] object-cover rounded-lg transition-transform duration-500 group-hover:scale-105"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = "https://static.vecteezy.com/system/resources/previews/037/359/798/non_2x/loading-error-flat-icon-style-illustration-eps-10-file-vector.jpg";
                  }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </div>
              <div className="space-y-4">
                <h2 className="text-3xl font-bold text-indigo-800">{movieDetails.title}</h2>
                <p className="text-gray-700 leading-relaxed">{movieDetails.overview}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default UserRatings; 