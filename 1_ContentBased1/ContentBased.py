'''
This work is prepared for the SDSC course titled
'Introduction to the Application of Data Science in Recommender Systems'
Data reference: https://www.kaggle.com/tmdb/tmdb-movie-metadata
'''

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.feature_extraction.text import TfidfVectorizer

movies_df = pd.read_csv("tmdb_5000_movies.csv") # read movie data
movies_df['overview'] = movies_df['overview'].fillna('') # fill empty entries

vectorizer = TfidfVectorizer() # initialize a tf-idf transformer

# apply tf-idf to 'overview' information; format - Compressed Sparse Row format
vectorizer_matrix = vectorizer.fit_transform(movies_df['overview'].tolist())

query_movie = 'Avatar' # specify a movie
#query_movie = random.choice(movies_df['original_title'].tolist()) # randomly choose a movie

# transform the query movie
query_tf_idf = vectorizer.transform([movies_df[movies_df['original_title']==query_movie].iloc[0]['overview']])

scores = cosine_similarity(query_tf_idf, vectorizer_matrix) # check content difference
scores_index = scores.ravel().argsort()[-15:][::-1] # sort from high similarity to low

print('Query:', query_movie)
for i in range(1, len(scores_index)):
    print('Recommend', i, movies_df.iloc[scores_index[i]]['original_title'])

