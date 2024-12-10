import numpy as np
import pandas as pd
from utils import euclidean, cosine_similarity, pearson_correlation, hamming_distance
import time

class MovieRecommender:
    def __init__(self, k_neighbors=30, n_recommendations=40, distance_metric='cosine'):
        self.k_neighbors = k_neighbors
        self.n_recommendations = n_recommendations
        self.distance_metrics = {
            'euclidean': euclidean,
            'cosine': cosine_similarity,
            'pearson': pearson_correlation,
            'hamming': hamming_distance
        }
        self.set_distance_metric(distance_metric)
        
    def set_distance_metric(self, metric_name):
        """Set the distance metric to use."""
        if metric_name not in self.distance_metrics:
            raise ValueError(f"Unsupported distance metric: {metric_name}")
        self.current_metric = metric_name
        self.distance_func = self.distance_metrics[metric_name]
        
    def load_main_data(self):
        """Load and preprocess the main MovieLens dataset."""
        data = pd.read_csv("movielens.txt", sep="\t", 
                          names=["user_id", "movie_id", "rating", "title", "genre", 
                                "age", "gender", "occupation"],
                          low_memory=False)
        
        numeric_cols = ['user_id', 'movie_id', 'rating']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data.dropna(subset=numeric_cols, inplace=True)
        data = data[['user_id', 'movie_id', 'rating', 'title']]
        data = data.groupby(['user_id', 'movie_id', 'title'], 
                          as_index=False).agg({'rating': 'mean'})
        return data
    
    def calculate_user_similarity(self, user_movie_matrix):
        """Calculate user similarity matrix using the selected distance metric."""
        n_users = len(user_movie_matrix)
        similarity_matrix = np.zeros((n_users, n_users))
        
        for i in range(n_users):
            for j in range(i, n_users):
                if self.current_metric in ['euclidean', 'hamming']:
                    # For distance metrics, convert to similarity
                    similarity = 1 / (1 + self.distance_func(
                        user_movie_matrix.iloc[i], user_movie_matrix.iloc[j]))
                else:
                    # For similarity metrics
                    similarity = self.distance_func(
                        user_movie_matrix.iloc[i], user_movie_matrix.iloc[j])
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def create_user_item_matrix(self, data):
        """Create user-item matrix and calculate user similarities."""
        user_movie_matrix = data.pivot(index='user_id', 
                                     columns='movie_id', 
                                     values='rating').fillna(0)
        
        user_similarity = self.calculate_user_similarity(user_movie_matrix)
        return user_movie_matrix, user_similarity
    
    def load_user_data(self, user_id, data_type='train'):
        """Load individual user data from files."""
        try:
            filename = f"{data_type}_{user_id}.txt"
            data = pd.read_csv(filename, sep="\t",
                             names=["user_id", "movie_id", "rating", "title", 
                                   "genre", "age", "gender", "occupation"],
                             low_memory=False)
            
            for col in ['user_id', 'movie_id', 'rating']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data.dropna(subset=['user_id', 'movie_id', 'rating'], inplace=True)
            
            # data = data[['user_id', 'movie_id', 'rating']]
            # data = data.groupby(['user_id', 'movie_id'], 
            #                   as_index=False).agg({'rating': 'mean'})
            print(data.columns)
            return data
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return pd.DataFrame(columns=["user_id", "movie_id", "rating"])

    def recommend_movies(self, user_id, user_movie_matrix, user_similarity, data):
        """Generate movie recommendations for a specific user."""
        user_idx = user_id - 1
        similarity_scores = user_similarity[user_idx]
        
        similar_users = np.argsort(similarity_scores)[-self.k_neighbors-1:-1][::-1]
        user_ratings = user_movie_matrix.iloc[user_idx]
        unrated_movies = user_ratings[user_ratings == 0].index
        
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
        
        recommended_movies = sorted(movie_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:self.n_recommendations]
        recommended_movie_ids = [movie[0] for movie in recommended_movies]
        
        recommended_details = data[data['movie_id'].isin(recommended_movie_ids)][['movie_id', 'title']].drop_duplicates()
        return recommended_details
    
    def evaluate_recommendations(self, recommended_movies, actual_ratings):
        """Evaluate recommendation performance using precision, recall, and F1."""
        recommended_set = set(recommended_movies['movie_id'])
        actual_set = set(actual_ratings['movie_id'])
        
        true_positives = len(recommended_set.intersection(actual_set))
        false_positives = len(recommended_set - actual_set)
        false_negatives = len(actual_set - recommended_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1

def main():
    # Initialize results storage
    results = []
    
    # Test different distance metrics
    distance_metrics = ['cosine']
    user_ids = ['a', 'b', 'c']
    user_numeric_ids = [405, 655, 13]  # Corresponding numeric IDs
    
    for metric in distance_metrics:
        print(f"\nTesting with {metric} distance metric:")
        print("=" * 50)
        
        # Initialize recommender with current metric
        recommender = MovieRecommender(distance_metric=metric)
        
        # Load main dataset
        print(f"Loading main dataset...")
        main_data = recommender.load_main_data()
        
        # Create user-item matrix
        print(f"Creating user-item matrix...")
        user_movie_matrix, user_similarity = recommender.create_user_item_matrix(main_data)
        
        # Process each test user
        for letter_id, numeric_id in zip(user_ids, user_numeric_ids):
            print(f"\nProcessing User {letter_id} (ID: {numeric_id}):")
            
            # Get recommendations
            recommendations = recommender.recommend_movies(
                numeric_id, user_movie_matrix, user_similarity, main_data
            )
            
            # Evaluate on validation set
            valid_ratings = recommender.load_user_data(letter_id, 'valid')
            if not valid_ratings.empty:
                v_precision, v_recall, v_f1 = recommender.evaluate_recommendations(
                    recommendations, valid_ratings
                )
                print(f"Validation Metrics - Precision: {v_precision:.4f}, Recall: {v_recall:.4f}, F1: {v_f1:.4f}")
            
            # Evaluate on test set
            test_ratings = recommender.load_user_data(letter_id, 'test')
            if not test_ratings.empty:
                t_precision, t_recall, t_f1 = recommender.evaluate_recommendations(
                    recommendations, test_ratings
                )
                print(f"Test Metrics - Precision: {t_precision:.4f}, Recall: {t_recall:.4f}, F1: {t_f1:.4f}")
                
                # Store results
                results.append({
                    'metric': metric,
                    'user': letter_id,
                    'validation_f1': v_f1,
                    'test_f1': t_f1
                })
    
    # Save results to file
    with open('results.txt', 'w') as f:
        f.write("Recommendation System Evaluation Results\n")
        f.write("======================================\n\n")
        
        for result in results:
            f.write(f"Distance Metric: {result['metric']}\n")
            f.write(f"User: {result['user']}\n")
            f.write(f"Validation F1 Score: {result['validation_f1']:.4f}\n")
            f.write(f"Test F1 Score: {result['test_f1']:.4f}\n")
            f.write("-" * 40 + "\n\n")

if __name__ == "__main__":
    main()