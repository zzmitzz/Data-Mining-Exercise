import { useState, useEffect } from 'react';
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
  const navigate = useNavigate();

  // Fetch users on component mount
  useEffect(() => {
    fetchUsers();
  }, []);

  // Fetch user ratings when a user is selected
  useEffect(() => {
    if (selectedUser) {
      fetchUserRatings(selectedUser);
      fetchRecommendations(selectedUser);
    }
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

  const fetchMovieDetails = async (movieId) => {
    setIsLoadingMovieDetails(true);
    try {
      const response = await fetch(`http://localhost:8000/movies/${movieId}`);
      if (!response.ok) throw new Error('Failed to fetch movie details');
      const data = await response.json();
      setMovieDetails(data["movie"]);
    } catch (error) {
      console.error('Error fetching movie details:', error);
    } finally {
      setIsLoadingMovieDetails(false);
    }
  };

  const handleMovieClick = (movie) => {
    setSelectedMovie(movie);
    fetchMovieDetails(movie.movieId);
  };

  const MovieCard = ({ movie, onClick }) => (
    <div 
      className="movie-card cursor-pointer p-4 bg-white rounded-xl shadow-sm hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1"
      onClick={() => onClick(movie)}
    >
      <div className="text-center">
        <h3 className="text-xl font-bold text-gray-800 mb-2">id: {movie.movieId}</h3>
        <div className="flex items-center justify-center gap-1">
          {[...Array(5)].map((_, index) => (
            <span key={index} className={`text-2xl ${index < movie.rating ? 'text-yellow-400' : 'text-gray-300'}`}>
              ★
            </span>
          ))}
        </div>
        <span className="text-gray-600 mt-1 block">({movie.rating}/5)</span>
      </div>
    </div>
  );

  const LoadingSpinner = () => (
    <div className="flex justify-center items-center p-8">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-8">
      <button 
        onClick={() => navigate('/')}
        className="mb-8 px-6 py-2.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors duration-300 shadow-sm hover:shadow-md flex items-center gap-2"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M9.707 14.707a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 1.414L7.414 9H15a1 1 0 110 2H7.414l2.293 2.293a1 1 0 010 1.414z" clipRule="evenodd" />
        </svg>
        Back to Main
      </button>

      <div className="flex gap-8">
        {/* Left Panel */}
        <div className="w-2/3">
          <div className="mb-8 bg-white p-6 rounded-xl shadow-sm">
            <label className="block text-xl font-semibold mb-3 text-gray-800">Select User</label>
            {isLoadingUsers ? (
              <LoadingSpinner />
            ) : (
              <select 
                value={selectedUser || ''} 
                onChange={(e) => setSelectedUser(e.target.value)}
                className="w-full p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all duration-300 appearance-none bg-white text-gray-700"
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
                <h2 className="text-2xl font-bold mb-6 text-gray-800">User's Rated Movies</h2>
                {isLoadingRatings ? (
                  <LoadingSpinner />
                ) : (
                  <div className="flex overflow-x-auto gap-6 pb-6 scrollbar-hide">
                    {userRatings.map(movie => (
                      <MovieCard 
                        key={movie} 
                        movie={movie} 
                        onClick={handleMovieClick}
                      />
                    ))}
                  </div>
                )}
              </div>

              <div>
                <h2 className="text-2xl font-bold mb-6 text-gray-800">Recommended Movies</h2>
                {isLoadingRecommendations ? (
                  <LoadingSpinner />
                ) : (
                  <div className="flex overflow-x-auto gap-6 pb-6 scrollbar-hide">
                    {recommendations.map(movie => (
                      <MovieCard 
                        key={movie} 
                        movie={movie} 
                        onClick={handleMovieClick}
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
            <div className="bg-white p-8 rounded-xl shadow-lg sticky top-8">
              <LoadingSpinner />
            </div>
          ) : movieDetails && (
            <div className="bg-white p-8 rounded-xl shadow-lg sticky top-8">
              <div className="relative overflow-hidden rounded-lg mb-6">
                <img 
                  src={`https://image.tmdb.org/t/p/w500${movieDetails.poster_path}`}
                  alt={movieDetails.title} 
                  className="w-full h-[500px] object-cover rounded-lg"
                  onError={(e) => {
                    e.target.onerror = null; // Prevent infinite loop
                    e.target.src = 'https://via.placeholder.com/500x750?text=No+Image+Available';
                  }}
                />
              </div>
              <div className="space-y-4">
                <h2 className="text-3xl font-bold text-gray-800">{movieDetails.title}</h2>
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