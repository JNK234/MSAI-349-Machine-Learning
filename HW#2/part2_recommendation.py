import numpy as np
import pandas as pd
from utils import cosine_similarity, hamming_distance, euclidean

class MovieRecommender:
    def __init__(self, k_neighbors=30, n_recommendations=40):
        self.k_neighbors = k_neighbors
        self.n_recommendations = n_recommendations
        
    def load_main_data(self):
        """Load and preprocess the main MovieLens dataset."""
        data = pd.read_csv("movielens.txt", sep="\t", 
                          names=["user_id", "movie_id", "rating", "title", "genre", 
                                "age", "gender", "occupation"],
                          low_memory=False)
        
        # Convert numeric columns
        numeric_cols = ['user_id', 'movie_id', 'rating']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Clean data
        data.dropna(subset=numeric_cols, inplace=True)
        data = data[['user_id', 'movie_id', 'rating', 'title']]
        data = data.groupby(['user_id', 'movie_id', 'title'], 
                          as_index=False).agg({'rating': 'mean'})
        return data
    
    def create_user_item_matrix(self, data):
        """Create user-item matrix and calculate user similarities."""
        user_movie_matrix = data.pivot(index='user_id', 
                                     columns='movie_id', 
                                     values='rating').fillna(0)
        
        # Calculate user similarity matrix
        norms = np.linalg.norm(user_movie_matrix, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_matrix = user_movie_matrix / norms[:, np.newaxis]
        user_similarity = np.dot(normalized_matrix, normalized_matrix.T)
        
        return user_movie_matrix, user_similarity
    
    def load_user_data(self, user_id):
        """Load individual user data from training files."""
        try:
            data = pd.read_csv(f"train_{user_id}.txt", sep="\t",
                             names=["user_id", "movie_id", "rating", "title", 
                                   "genre", "age", "gender", "occupation"],
                             low_memory=False)
            
            # Convert and clean numeric columns
            for col in ['user_id', 'movie_id', 'rating']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data.dropna(subset=['user_id', 'movie_id', 'rating'], inplace=True)
            
            # Select and aggregate relevant columns
            data = data[['user_id', 'movie_id', 'rating']]
            data = data.groupby(['user_id', 'movie_id'], 
                              as_index=False).agg({'rating': 'mean'})
            return data
        except FileNotFoundError:
            print(f"File train_{user_id}.txt not found.")
            return pd.DataFrame(columns=["user_id", "movie_id", "rating"])
    
    def recommend_movies(self, user_id, user_movie_matrix, user_similarity, data):
        """Generate movie recommendations for a specific user."""
        user_idx = user_id - 1
        similarity_scores = user_similarity[user_idx]
        
        # Get top K similar users
        similar_users = np.argsort(similarity_scores)[-self.k_neighbors-1:-1][::-1]
        
        # Get movies not rated by the target user
        user_ratings = user_movie_matrix.iloc[user_idx]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Calculate predicted ratings
        movie_scores = {}
        for movie_id in unrated_movies:
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user in similar_users:
                sim_score = similarity_scores[similar_user]
                rating = user_movie_matrix.iloc[similar_user][movie_id]
                
                if rating > 0:
                    weighted_sum += sim_score * rating
                    similarity_sum += sim_score
            
            if similarity_sum > 0:
                movie_scores[movie_id] = weighted_sum / similarity_sum
        
        # Get top M recommendations
        recommended_movies = sorted(movie_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:self.n_recommendations]
        recommended_movie_ids = [movie[0] for movie in recommended_movies]
        
        # Get movie details
        recommended_details = data[data['movie_id'].isin(recommended_movie_ids)][['movie_id', 'title']].drop_duplicates()
        return recommended_details
    
    def evaluate_recommendations(self, recommended_movies, actual_ratings):
        """Evaluate recommendation performance using precision, recall, and F1."""
        recommended_set = set(recommended_movies['movie_id'])
        actual_set = set(actual_ratings['movie_id'])
        
        # Calculate metrics
        true_positives = len(recommended_set.intersection(actual_set))
        false_positives = len(recommended_set - actual_set)
        false_negatives = len(actual_set - recommended_set)
        
        # Calculate precision, recall, and F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1

def main():
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Load main dataset
    print("Loading main dataset...")
    main_data = recommender.load_main_data()
    
    # Create user-item matrix
    print("Creating user-item matrix...")
    user_movie_matrix, user_similarity = recommender.create_user_item_matrix(main_data)
    
    # Test users
    user_ids = ['a', 'b', 'c']
    user_numeric_ids = [405, 655, 13]  # Corresponding numeric IDs
    
    print("\nGenerating recommendations and evaluating performance...")
    print("-" * 70)
    
    # Process each test user
    for letter_id, numeric_id in zip(user_ids, user_numeric_ids):
        print(f"\nProcessing User {letter_id} (ID: {numeric_id}):")
        print("-" * 40)
        
        # Get recommendations
        recommendations = recommender.recommend_movies(
            numeric_id, user_movie_matrix, user_similarity, main_data
        )
        print(f"\nTop {recommender.n_recommendations} Recommended Movies:")
        print(recommendations.to_string(index=False))
        
        # Load actual ratings for evaluation
        actual_ratings = recommender.load_user_data(letter_id)
        
        if not actual_ratings.empty:
            # Evaluate recommendations
            precision, recall, f1 = recommender.evaluate_recommendations(
                recommendations, actual_ratings
            )
            
            print(f"\nPerformance Metrics:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
        else:
            print(f"\nNo actual ratings found for evaluation.")
            
        print("\n" + "=" * 70)

if __name__ == "__main__":
    main()