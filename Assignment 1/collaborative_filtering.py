import pandas as pd
import numpy as np
import multiprocessing
import random

"""The u.data dataset contains the ranking assigned by the users 
of a streaming platform to the movies available on the platform."""
# Importing dataset: paste your path to u.data in the following line:
path = "u.data"
df = pd.read_table(path, sep="\t", names=["UserID", "MovieID", "Rating", "Timestamp"])
print(df.head())  # to check the correct import of the dataset

df = df.pivot_table(index='UserID', columns='MovieID', values='Rating')

def cosine_similarity(u, v):
    u = np.array(u)
    v = np.array(v)

    dot_product = np.dot(u, v)
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)

    if magnitude_u == 0 or magnitude_v == 0:
        return 0

    result = dot_product / (magnitude_u * magnitude_v)
    return result

def similarity_matrix(matrix, k, axis):
    similarity_dict = {}
    matrix.fillna(0, inplace=True)

    if axis == 1:
        matrix = matrix.T

        # Convert the DataFrame to a NumPy array for faster computation
    matrix_np = matrix.to_numpy()

    # Loop through each pair of entities (users or items) to calculate cosine similarity
    for i in range(len(matrix_np)):
        similarities = []
        for j in range(len(matrix_np)):
            if i != j:
                similarity = cosine_similarity(matrix_np[i], matrix_np[j])
                similarities.append((j, similarity))

        # Sort the similarity scores and keep the top k
        top_k_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        similarity_dict[i] = [(item, sim) for item, sim in top_k_similarities]

    return similarity_dict

def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    # TO DO: retrieve the top k most similar users for the target user
    similar_users = sorted(user_similarity[user_id], key=lambda x: x[1], reverse=True)[:k]
    # TO DO: implement user-based collaborative filtering according to the formula discussed
    #        during the lecture (reported in the PDF attached to the assignment)
    numerator = 0
    denominator = 0

    for similar_user, similarity_score in similar_users:
        user_rating = user_item_matrix.loc[similar_user, movie_id]

        # Skip users with missing (NaN) or zero ratings
        if not np.isnan(user_rating) and user_rating != 0:
            numerator += similarity_score * user_rating
            denominator += similarity_score

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator
    return predicted_rating


def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):
    # TO DO: retrieve the topk most similar users for the target item
    similar_items = sorted(item_similarity[movie_id], key=lambda x: x[1], reverse=True)[:k]
    # TO DO: implement item-based collaborative filtering according to the formula discussed
    #        during the lecture (reported in the PDF attached to the assignment)
    numerator = 0
    denominator = 0

    for similar_item, similarity_score in similar_items:
        user_rating = user_item_matrix.loc[user_id, similar_item]
        if not np.isnan(user_rating) and user_rating != 0:
            numerator += similarity_score * user_rating
            denominator += similarity_score

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


if __name__ == '__main__':
    user_id = 13
    movie_id = 100
    user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)

    u_predicted_rating = user_based_cf(user_id, movie_id, user_similarity_matrix, user_item_matrix=df, k=5)
    print(
        f"predicted user {user_id} rating for movie {movie_id}, according to user-based collaborative filtering is: {u_predicted_rating:.2f}")

    i_predicted_rating = item_based_cf(user_id, movie_id, item_similarity_matrix, user_item_matrix=df, k=5)
    print(
        f"predicted user {user_id} rating for movie {movie_id}, according to item-based collaborative filtering is: {i_predicted_rating:.2f}")
