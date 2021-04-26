'''
This work is prepared for the SDSC course titled
'Introduction to the Application of Data Science in Recommender Systems'
Data reference: https://grouplens.org/datasets/movielens/
'''

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


path = '' # data directory

movies_df = pd.read_csv(path + 'movies.csv', usecols=['movieId','title']) # 9742 movies
rating_df=pd.read_csv(path + 'ratings.csv', usecols=['userId', 'movieId', 'rating']) # 100836 ratings

df = pd.merge(rating_df, movies_df, on='movieId') # add movie titles to the rating table

# remove movies (rows) without title; in this case, it removes nothing
movie_rating = df.dropna(axis = 0, subset = ['title'])


## check the total number of ratings for each movie - optional 1 (slow)
#rating_movies_unique = list(set(movie_rating['title'].tolist())) # unique movie titles
#rating_movies = movie_rating['title'].tolist() # all movies titles containing duplicates
#rows = [[rating_movies_unique[i], rating_movies.count(rating_movies_unique[i])] for i in range(len(rating_movies_unique))] # count ratings for each movie
#movie_ratingcount = pd.DataFrame(rows, columns=['title', 'total_rating_count']) # number of ratings for each movies


# check the total number of ratings for each movie - optional 2 (fast)
movie_ratingcount = movie_rating.groupby(by = ['title']) # group ratings for each movie
movie_ratingcount = movie_ratingcount['rating'] # specify the column for counting
movie_ratingcount = movie_ratingcount.count() # get the number of ratings for each movie; dataframe with movie titles as indices
movie_ratingcount = movie_ratingcount.reset_index() # add numerical index to the dateframe
movie_ratingcount = movie_ratingcount.rename(columns = {'rating': 'total_rating_count'}) # rename the column


movie_rating = movie_rating.merge(movie_ratingcount, how = 'left') # add movies counts to the rating table

min_ratings = 50 # we only recommend movies with at least 50 ratings
movie_rating_popular = movie_rating.query('total_rating_count >= @min_ratings') # filter

# convert the ratings to tabular format
movie_user_matrix = movie_rating_popular.pivot_table(index='title',columns='userId',values='rating')
movie_user_matrix = movie_user_matrix.fillna(0) # fill empty entries

movie_user_matrix_csr = csr_matrix(movie_user_matrix.values) # convert to Compressed Sparse Row format

# Unsupervised learner for implementing neighbor searches
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_user_matrix_csr) # fit data

query = np.random.choice(movie_user_matrix.shape[0]) # randomly choose a movie
#query = 37 # <<Avengers>>

# look for similar movies
distances, indices = model_knn.kneighbors(movie_user_matrix.iloc[query,:].values.reshape(1, -1), n_neighbors = 15)

# print search result
for i in range(0, indices.shape[1]):
    if i == 0:
        print('Query movie:', movie_user_matrix.index[query])
    else:
        print(i, movie_user_matrix.index[indices[0,i]], '. Distance =', distances[0,i])

