'''
This work is prepared for the SDSC course titled
'Introduction to the Application of Data Science in Recommender Systems'
Data reference: https://grouplens.org/datasets/movielens/
'''

import pandas as pd
import numpy as np
import random
from scipy.sparse.linalg import svds

path = '' # data directory

movies_df = pd.read_csv(path + 'movies.csv', usecols=['movieId','title']) # 9742 movies

# 100836 ratings; the ratings may not cover all movies above
ratings_df=pd.read_csv(path + 'ratings.csv', usecols=['userId', 'movieId', 'rating'])

# create a user-movie matrix (rating matrix)
user_movie_matrix = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)

ratings_matrix = user_movie_matrix.values # convert to a numerical matrix


## rating prediction. normalization optional 1
#user_ratings_mean = np.mean(ratings_matrix, axis = 1) # mean rating for each customer
#ratings_matrix_revised = ratings_matrix - user_ratings_mean.reshape(-1, 1) # normalize
#U, sigma, Vt = svds(ratings_matrix_revised, k = 50) # SVD; feature length = 50
#sigma = np.diag(sigma)
#predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1) # predict


# rating prediction. normalization optional 2 (with normalization)
ratings_matrix_revised = ratings_matrix/5 # normalize
U, sigma, Vt = svds(ratings_matrix_revised, k = 5) # SVD; feature length = 50
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)*5 # predict


# Making Movie Recommendations
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns = user_movie_matrix.columns) # convert to dataframe

userID = 100 # specify a customer
#userID = random.choice(list(user_movie_matrix.index)) # randomly choose a customer
original_ratings_df = ratings_df
num_recommendations = 10 # make 10 recommendations

all_predictions = predicted_ratings_df.iloc[userID - 1].sort_values(ascending=False) # UserID starts at 1

movies_watched = original_ratings_df[original_ratings_df.userId == userID] # collect movies watched by this customer
movies_watched = movies_watched.merge(movies_df) # add movies titles

recommendations = movies_df[~movies_df['movieId'].isin(movies_watched['movieId'])] # movies not watched yet
all_predictions_df = pd.DataFrame(all_predictions) # convert to dataframe
all_predictions_df = all_predictions_df.reset_index() # reset index
recommendations = recommendations.merge(all_predictions_df, how = 'left') # add predicted ratings

# change column name: from UserID to Prediction
recommendations = recommendations.rename(columns = {userID - 1: 'Predictions'})
recommendations = recommendations.sort_values('Predictions', ascending = False) # from good to bad
recommendations = recommendations.iloc[:num_recommendations, :-1] # extract top recommendations

print('Recommended Movies for User', userID)
[print(c+1, i) for c,i in enumerate(recommendations['title'].tolist())]



