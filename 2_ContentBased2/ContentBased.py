'''
This work is prepared for the SDSC course titled
'Introduction to the Application of Data Science in Recommender Systems'
Data reference: https://tianchi.aliyun.com/competition/entrance/231575/information
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time
import numpy as np
import random

df = pd.read_excel('Item_Info.xlsx') # read the product information

#itemID = 8499 # specify a product
itemID = random.choice(df['ID'].tolist()) # randomly choose a product


cat_ID = df[df['ID']==itemID]['cat'].values[0] # get the category number
df = df[df['cat']==cat_ID] # get all products under this category
df = df.reset_index(drop=True) # reset index
df = df.drop(['Unnamed: 0'], axis=1)

print('Number of items in the same category', df.shape[0])

start = time.time()
tf = TfidfVectorizer() # initialize a tf-idf transformer
tfidf_matrix = tf.fit_transform(df['desc']) # apply tf-idf to all product descriptions

# build a product-to-product similarity matrix
similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

end = time.time()
print('Time used for tf-idf: ', np.round(end - start,2), 's.')

# sort from high similarity to low
similar_indices = similarity_matrix[df.index[df['ID']==itemID][0]].argsort()[:-13:-1]

similar_items_IDs = [df.iloc[i]['ID'] for i in similar_indices] # IDs of similar product





