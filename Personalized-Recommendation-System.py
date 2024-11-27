import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample user-item rating data (User, Movie, Rating)
data = {
    'User': ['Alice', 'Alice', 'Bob', 'Bob', 'Carol'],
    'Movie': ['Movie1', 'Movie2', 'Movie1', 'Movie3', 'Movie2'],
    'Rating': [5, 3, 4, 5, 4]
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Create a pivot table (user-item matrix)
pivot_table = df.pivot(index='User', columns='Movie', values='Rating').fillna(0)

# Calculate the cosine similarity between users
user_similarity = cosine_similarity(pivot_table)

# Convert the similarity matrix to a DataFrame for easy viewing
user_similarity_df = pd.DataFrame(user_similarity, index=pivot_table.index, columns=pivot_table.index)

# Function to get movie recommendations for a given user
def get_recommendations(user, num_recommendations=2):
    # Get the most similar users to the given user
    similar_users = user_similarity_df[user].sort_values(ascending=False).index[1:]
    
    recommended_movies = set()

    # Loop through similar users and recommend movies they've rated highly
    for similar_user in similar_users:
        # Get the movies that the similar user has rated
        similar_user_ratings = pivot_table.loc[similar_user]
        
        # Recommend movies that the similar user has rated and the original user hasn't yet rated
        for movie, rating in similar_user_ratings.items():
            if rating >= 4 and movie not in pivot_table.loc[user]:
                recommended_movies.add(movie)
        
        if len(recommended_movies) >= num_recommendations:
            break
    
    return list(recommended_movies)

# Example: Get recommendations for Alice
user_to_recommend = 'Alice'
recommended_movies = get_recommendations(user_to_recommend, num_recommendations=2)

print(f"Recommended movies for {user_to_recommend}: {recommended_movies}")
