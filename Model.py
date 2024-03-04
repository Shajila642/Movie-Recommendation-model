!pip install surprise
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the TMDB 5000 Movie Dataset (replace 'tmdb_5000_movies.csv' with your actual file path)
file_path = '/content/tmdb_5000_movies.csv'
df = pd.read_csv(file_path)

# Keep only the relevant columns (user, movie, rating)
ratings_df = df[['id', 'title', 'vote_average']]

# Rename columns for Surprise compatibility
ratings_df.columns = ['userId', 'title', 'rating']

# Create a Surprise Reader object
reader = Reader(rating_scale=(0, 10))  # Assuming the ratings are on a 0-10 scale

# Load the dataset from the DataFrame
data = Dataset.load_from_df(ratings_df[['userId', 'title', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Choose a collaborative filtering algorithm
sim_options = {
    'name': 'cosine',  # Use cosine similarity
    'user_based': False,  # Item-based collaborative filtering
}
knn_model = KNNBasic(sim_options=sim_options)

# Train the model
knn_model.fit(trainset)

# Make predictions and evaluate
predictions = knn_model.test(testset)
accuracy.rmse(predictions)

# Function to get movie recommendations based on a given movie
def get_movie_recommendations(movie_title, top_n=5):
    movie_id = trainset.to_inner_iid(movie_title)
    similar_movies = knn_model.get_neighbors(movie_id, k=top_n)

    # Convert movie IDs back to movie titles
    recommended_movies = [trainset.to_raw_iid(movie_id) for movie_id in similar_movies]
    return recommended_movies

# Example: Recommend movies similar to 'Inception'
given_movie = 'Inception'
recommendations = get_movie_recommendations(given_movie)

# Print the recommendations
print(f"Recommended movies similar to '{given_movie}':")
for movie_title in recommendations:
    print(movie_title)
