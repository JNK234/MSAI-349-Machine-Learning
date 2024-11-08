# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity


# def user_item_matrix(data):
#     user_movie_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
#     user_similarity = cosine_similarity(user_movie_matrix)
#     user_similarity = custom_cosine_similarity(user_movie_matrix)
#     return user_movie_matrix, user_similarity

# # Load and Preprocess Data
# def load_main_data():
#     data = pd.read_csv("movielens.txt", sep="\t", names=["user_id", "movie_id", "rating", "title", "genre", "age", "gender", "occupation"], low_memory=False)
#     data['user_id'] = pd.to_numeric(data['user_id'], errors='coerce')
#     data['movie_id'] = pd.to_numeric(data['movie_id'], errors='coerce')
#     data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
#     data.dropna(subset=['user_id', 'movie_id', 'rating'], inplace=True)
#     data = data[['user_id', 'movie_id', 'rating', 'title']]
#     data = data.groupby(['user_id', 'movie_id', 'title'], as_index=False).agg({'rating': 'mean'})
#     return data


# # Collaborative Filtering Recommendation Function
# def movies_recommended(user_id, user_movie_matrix, user_similarity, data, K, M):
#     user_ids = user_id - 1 
#     similarity_scores = user_similarity[user_ids]
#     similar_users = np.argsort(similarity_scores)[-K-1:-1][::-1]        # top K similar users
#     user_ratings = user_movie_matrix.iloc[user_ids]
#     movies_notrated = user_ratings[user_ratings == 0].index
#     movie_scores = {}
#     for movie_id in movies_notrated:
#         total_score = 0
#         sum = 0
#         for similar_user in similar_users:
#             similarity_score = similarity_scores[similar_user]
#             rating = user_movie_matrix.iloc[similar_user][movie_id]
#             if rating > 0:  
#                 total_score += similarity_score * rating
#                 sum += similarity_score
#         if sum > 0:
#             movie_scores[movie_id] = total_score / sum
#     recommended_movie_ids = sorted(movie_scores, key=movie_scores.get, reverse=True)[:M]
#     recommended_movies = data[data['movie_id'].isin(recommended_movie_ids)][['movie_id', 'title']].drop_duplicates()
#     return recommended_movies


# def custom_cosine_similarity(matrix):
#     norms = np.linalg.norm(matrix, axis=1)
#     norms[norms == 0] = 1
#     normalized_matrix = matrix / norms[:, np.newaxis]  
#     similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
#     return similarity_matrix

# # Evaluate recommendations
# def evaluate_recommendations(recommended_movies, actual_ratings):
#     actual_set = set(actual_ratings['movie_id'])
#     recommended_set = set(recommended_movies['movie_id'])
#     tp = len(recommended_set.intersection(actual_set))  # True positives
#     fp = len(recommended_set - actual_set)  # False positives
#     fn = len(actual_set - recommended_set)  # False negatives
#     precision = tp / (tp + fp) if tp + fp > 0 else 0
#     recall = tp / (tp + fn) if tp + fn > 0 else 0
#     f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
#     return precision, recall, f1

# def load_movie_titles(data):
#     return data[['movie_id', 'title']].drop_duplicates().set_index('movie_id')

# def user_data(user_id):
#     try:
#         data = pd.read_csv(f"train_{user_id}.txt", sep="\t", names=["user_id", "movie_id", "rating", "title", "genre", "age", "gender", "occupation"], low_memory=False)
#         data['user_id'] = pd.to_numeric(data['user_id'], errors='coerce')
#         data['movie_id'] = pd.to_numeric(data['movie_id'], errors='coerce')
#         data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
#         data.dropna(subset=['user_id', 'movie_id', 'rating'], inplace=True)
#         data = data[['user_id', 'movie_id', 'rating']]
#         data = data.groupby(['user_id', 'movie_id'], as_index=False).agg({'rating': 'mean'})
#         return data
#     except FileNotFoundError:
#         print(f"File train_{user_id}.txt not found.")
#         return pd.DataFrame(columns=["user_id", "movie_id", "rating"])

# # Main Program
# if __name__ == "__main__":
#     main_data = load_main_data()
#     user_movie_matrix, user_similarity = user_item_matrix(main_data)
#     movie_titles = load_movie_titles(main_data)

#     user_ids = ['a', 'b', 'c']  # Example users
#     user_ac_ids = [405, 655, 13]  # Numeric IDs corresponding to users a, b, c


#     K = 30  # Number of similar users
#     M = 40 # Number of recommendations
#     actual_ratings = pd.concat([user_data(user_id) for user_id in user_ids])


#     for i, user_id in enumerate(user_ids):
#         user_id_numeric = user_ac_ids[i]  # Use the correct numeric user ID
#         recommend_movies = movies_recommended(user_id_numeric, user_movie_matrix, user_similarity, main_data, K, M)
#         movie_titles = recommend_movies.reset_index(drop=True)
#         print(f"Recommended movies for User {user_id}:\n{movie_titles}\n")
#         actual_rating = actual_ratings[actual_ratings['user_id'] == user_id_numeric]
#         print(f"Actual ratings for User {user_id}:\n{actual_rating}\n")
        
#         if actual_rating.empty:
#             print(f"No actual ratings found for User {user_id}.\n")
#             continue
#         precision, recall, f1 = evaluate_recommendations(movie_titles, actual_rating)
        
#         print(f"Metrics for User {user_id}:")
#         print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
#         print("*********************\n\n")

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Function to create user-item matrix and calculate similarity
def user_item_matrix(data, filter_attributes=None):
    # Apply optional filters on the data based on attributes
    if filter_attributes:
        for attr, value in filter_attributes.items():
            data = data[data[attr] == value]

    user_movie_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    user_similarity = custom_cosine_similarity(user_movie_matrix)
    return user_movie_matrix, user_similarity

# Load and preprocess data
def load_main_data():
    data = pd.read_csv("movielens.txt", sep="\t", names=["user_id", "movie_id", "rating", "title", "genre", "age", "gender", "occupation"], low_memory=False)
    data[['user_id', 'movie_id', 'rating']] = data[['user_id', 'movie_id', 'rating']].apply(pd.to_numeric, errors='coerce')
    data.dropna(subset=['user_id', 'movie_id', 'rating'], inplace=True)
    return data

# Collaborative Filtering Recommendation Function with filters
def movies_recommended(user_id, user_movie_matrix, user_similarity, data, K, M):
    user_idx = user_id - 1 
    similarity_scores = user_similarity[user_idx]
    similar_users = np.argsort(similarity_scores)[-K-1:-1][::-1]
    
    user_ratings = user_movie_matrix.iloc[user_idx]
    movies_not_rated = user_ratings[user_ratings == 0].index
    movie_scores = {}

    for movie_id in movies_not_rated:
        total_score = sum_score = 0
        for similar_user in similar_users:
            similarity_score = similarity_scores[similar_user]
            rating = user_movie_matrix.iloc[similar_user][movie_id]
            if rating > 0:
                total_score += similarity_score * rating
                sum_score += similarity_score
        if sum_score > 0:
            movie_scores[movie_id] = total_score / sum_score

    recommended_movie_ids = sorted(movie_scores, key=movie_scores.get, reverse=True)[:M]
    return data[data['movie_id'].isin(recommended_movie_ids)][['movie_id', 'title']].drop_duplicates()

# Custom cosine similarity calculation
def custom_cosine_similarity(matrix):
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1
    normalized_matrix = matrix / norms[:, np.newaxis]
    return np.dot(normalized_matrix, normalized_matrix.T)

# Evaluation Function
def evaluate_recommendations(recommended_movies, actual_ratings):
    actual_set = set(actual_ratings['movie_id'])
    recommended_set = set(recommended_movies['movie_id'])
    tp = len(recommended_set.intersection(actual_set))
    fp = len(recommended_set - actual_set)
    fn = len(actual_set - recommended_set)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

# Main Program with flexible attribute filtering
if __name__ == "__main__":
    main_data = load_main_data()
    
    # Specify filters for recommendation (e.g., only for users aged 25-30 and female)
    filter_attributes = {'age': 25, 'gender': 'F'}
    user_movie_matrix, user_similarity = user_item_matrix(main_data, filter_attributes)
    
    K = 30  # Number of similar users
    M = 10  # Number of recommendations
    
    # Example user ID
    user_id_numeric = 405  
    recommended_movies = movies_recommended(user_id_numeric, user_movie_matrix, user_similarity, main_data, K, M)
    print(f"Recommended movies for User {user_id_numeric}:\n{recommended_movies}")


