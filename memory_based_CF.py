#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Sophia Bouchama

#--------------------------IMPORT LIBRARIES-----------------------------------#
import pandas
import numpy as np
from numpy import count_nonzero
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math 
#-----------------------------------------------------------------------------#

#---------------------------PRE PROCESSING------------------------------------#
#Load files
triplets_file = 'triplet.txt'
songs_metadata_file = 'song_metadata.csv'

#create a df
song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']
song_df_2 = pandas.read_csv(songs_metadata_file)
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")


#create a sorted data frame that lists the users with their number of occurrences
count_df = pandas.DataFrame(song_df_1.user_id.value_counts())
# join the count data frame back with the original data frame
new_index = count_df.merge(song_df_1[["user_id"]], left_index=True, right_on="user_id")
# output the original data frame in the order of the new index
sorted_df = song_df_1.reindex(new_index.index)


#Top 25 percent and bottom 25 percent of users
n = 25
top_25_users = sorted_df.head(int(len(sorted_df)*(n/100)))
bottom_25_users = sorted_df.tail(int(len(sorted_df)*(n/100)))

top_25_pivot = top_25_users.pivot(index='user_id', columns='song_id', values='listen_count')
top_25_ratings_df = top_25_pivot.fillna(0)
top_25_ratings_matrix = top_25_ratings_df.values

bottom_25_pivot = bottom_25_users.pivot(index='user_id', columns='song_id', values='listen_count')
bottom_25_ratings_df = bottom_25_pivot.fillna(0)
bottom_25_ratings_matrix = bottom_25_ratings_df.values


#Convert dataframe into a matrix
pivot1 = song_df_1.pivot(index='user_id', columns='song_id', values='listen_count')
ratings_df = pivot1.fillna(0)
ratings_matrix = ratings_df.values

#Calculate sparsity function
def calculate_sparsity(matrix):    
    sparsity = (count_nonzero(matrix)/matrix.size)*100
    return sparsity

#Print sparsity levels
print('Top 25% Sparsity:', calculate_sparsity(top_25_ratings_matrix))
print('Bottom 25% Sparsity:', calculate_sparsity(bottom_25_ratings_matrix), '\n')

#Split data into training and testing
train, test = train_test_split(top_25_ratings_matrix, test_size = 0.3)
#-----------------------------------------------------------------------------#

#Number of users and number of items
n_users = np.shape(train)[0]
n_items = np.shape(train.T)[0]

epsilon = 1e-100

#Weighting for IUF 
m_j = [np.count_nonzero(train[:,i]) for i in range(n_items)]
weighting = [math.log10(n_users/(i + epsilon)) for i in m_j]

#Pearson similarity function
def pearson_sim(x,y,w=1):
    #Calculate mean of vector x and vector y
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    
    #Calculate the numerator
    q = (x-x_bar)*(y-y_bar)*w
    numerator = sum(q)

    #Calculate the denominator
    a = ((x-x_bar)**2)*w
    b = math.sqrt(sum(a))
    
    c = ((y-y_bar)**2)*w
    d = math.sqrt(sum(c))
    denominator = b*d
    
    pearson = numerator/(denominator + epsilon)
    
    return pearson
    
#Cosine similarity function
def cosine_sim(x,y):
    dot_product = np.dot(x,y)
    mod_x = np.sqrt((x ** 2).sum())
    mod_y = np.sqrt((y ** 2).sum())
    cosine = dot_product / ((mod_x * mod_y) + epsilon)
    
    return cosine


#Similarity matrix function
def similarity_matrix(matrix, metric ='pearson', type='user', w=1):
    #Define size of the matrix depending on if user-user similarity or item-item similarity is being computed
    if type == 'user':
        x = n_users
    elif type == 'item':
        x = n_items
    if metric == 'pearson':
        #Loop through the empty matrix entering the Pearson similarity score
        B = np.zeros(shape=(x,x))
        for i in range(x):
            for j in range(i, x):
                value = pearson_sim(matrix[i,:], matrix[j,:], w)
                B[i,j] = value
                B[j,i] = value
    if metric == 'cosine':
        #Loop through the empty matrix entering the cosine similarity score
        B = np.zeros(shape=(x,x))
        for i in range(x):
            for j in range(i, x):
                value = cosine_sim(matrix[i,:], matrix[j,:])
                B[i,j] = value
                B[j,i] = value
                
    return B

#-----------------------------Prediction Functions----------------------------#
#Basic prediction function
def basic(ratings, similarity, type='user'):
    if type == 'user':
        pred = np.dot(similarity,ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = np.dot(ratings,similarity) / (np.array([np.abs(similarity).sum(axis=1)]) + epsilon) 
    
    return pred

#Topk prediction function
def topk(ratings, similarity, type='user', k=10):
    pred = np.zeros(shape=(n_users,n_items))
    if type == 'user':
        for i in range(n_users):
            #Find nearest k neighbours
            #Sort similarity score and select the top k
            top_k_users = [np.argsort(similarity[i])[:-1-k:-1]]
            for j in range(n_items):
                pred[i,j] = np.dot(similarity[i, :][top_k_users], (ratings[:, j][top_k_users])) / np.sum(np.abs(similarity[i, :][top_k_users]))
    if type == 'item':
        for j in range(n_items):
            #Find nearest k neighbours
            #Sort similarity score and select the top k
            top_k_items = [np.argsort(similarity[j])[:-1-k:-1]]
            for i in range(n_users):
                pred[i,j] = np.dot(similarity[j, :][top_k_items], (ratings[i, :][top_k_items].T)) / (np.sum(np.abs(similarity[j, :][top_k_items])) + epsilon)

    return pred

#nNo bias prediction function
def no_bias(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = np.mean(ratings, axis=1)
        #np.newaxis is used so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + (np.dot(similarity,ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T)
    elif type == 'item':
        mean_item_rating = np.mean(ratings, axis=0)
        ratings_diff = (ratings - mean_item_rating[np.newaxis, :])
        pred = mean_item_rating[np.newaxis, :] + ((np.dot(ratings_diff,similarity)) / (np.array([np.abs(similarity).sum(axis=1)]) + epsilon))
        
    return pred

#Top-k and No bias function
def topk_no_bias(ratings, similarity, type='user', k=10):
    pred = np.zeros(shape=(n_users,n_items))
    if type == 'user':
        mean_user_rating = np.mean(ratings, axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        for i in range(n_users):
            top_k_users = [np.argsort(similarity[i])[:-1-k:-1]]
            for j in range(n_items):
                pred[i,j] = np.dot(similarity[i, :][top_k_users], (ratings_diff[:, j][top_k_users])) / np.sum(np.abs(similarity[i, :][top_k_users]))
        pred = pred + mean_user_rating[:, np.newaxis]
    if type == 'item':
        mean_item_rating = np.mean(ratings, axis=0)
        ratings_diff = (ratings - mean_item_rating[np.newaxis, :])
        for j in range(n_items):
            top_k_items = [np.argsort(similarity[j])[:-1-k:-1]]
            for i in range(n_users):
                pred[i,j] =  np.dot((ratings_diff[i, :][top_k_items].T), (similarity[j, :][top_k_items])) / (np.sum(np.abs(similarity[j, :][top_k_items])) + epsilon)
        pred = pred + mean_item_rating[np.newaxis, :]
        
    return pred
#-----------------------------------------------------------------------------#

#----------------------Similarity Matrices------------------------------------#
user_pearson = similarity_matrix(train, metric='pearson', type='user')
user_pearson_weighting = similarity_matrix(train, metric='pearson', type='user', w=weighting)
user_cosine = similarity_matrix(train, metric='cosine', type='user')
item_pearson = similarity_matrix(train.T, metric='pearson', type='item')
item_cosine = similarity_matrix(train.T, metric='cosine', type='item')
#-----------------------------------------------------------------------------#

#---------------------(User Based) Prediction Functions-----------------------#
#User Based Pearson (without weighting)
prediction_basic_pearson = basic(train, user_pearson, type='user')
prediction_topk_pearson = topk(train, user_pearson, type='user')
prediction_no_bias_pearson = no_bias(train, user_pearson, type='user')
prediction_topk_no_bias_pearson = topk_no_bias(train, user_pearson, type='user')

#User Based Cosine
prediction_basic_cosine = basic(train, user_cosine, type='user')
prediction_topk_cosine = topk(train, user_cosine, type='user')
prediction_no_bias_cosine = no_bias(train, user_cosine, type='user')
prediction_topk_no_bias_cosine = topk_no_bias(train, user_cosine, type='user')

#User Based Weighted Pearson
prediction_basic_wpearson = basic(train, user_pearson_weighting, type='user')
prediction_topk_wpearson = topk(train, user_pearson_weighting, type='user')
prediction_no_bias_wpearson = no_bias(train, user_pearson_weighting, type='user')
prediction_topk_no_bias_wpearson = topk_no_bias(train, user_pearson_weighting, type='user')
#-----------------------------------------------------------------------------#

#---------------------(Item Based) Prediction Functions-----------------------#
#Item Based Pearson
prediction_basic_pearson = basic(train, item_pearson, type='item')
prediction_topk_pearson = topk(train, item_pearson, type='item')
prediction_no_bias_pearson = no_bias(train, item_pearson, type='item')
prediction_topk_no_bias_pearson = topk_no_bias(train, item_pearson, type='item')

#Item Based Cosine
prediction_basic_cosine = basic(train, item_cosine, type='item')
prediction_topk_cosine = topk(train, item_cosine, type='item')
prediction_no_bias_cosine = no_bias(train, item_cosine, type='item')
prediction_topk_no_bias_cosine = topk_no_bias(train, item_cosine, type='item')
#-----------------------------------------------------------------------------#

#------------------------------RMSE calculation-------------------------------#
def calculate_rmse(pred, actual):
    #Disregard non-zero terms
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    
    return math.sqrt(mean_squared_error(pred, actual))
#-----------------------------------------------------------------------------#

#---------------------------RMSE Scores---------------------------------------#
rmse_basic_pearson = calculate_rmse(prediction_basic_pearson, test)
rmse_topk_pearson = calculate_rmse(prediction_topk_pearson, test)
rmse_no_bias_pearson = calculate_rmse(prediction_no_bias_pearson, test)
rmse_topk_no_bias_pearson = calculate_rmse(prediction_topk_no_bias_pearson, test)

rmse_basic_cosine = calculate_rmse(prediction_basic_cosine, test)
rmse_topk_cosine = calculate_rmse(prediction_topk_cosine, test)
rmse_no_bias_cosine = calculate_rmse(prediction_no_bias_cosine, test)
rmse_topk_no_bias_cosine = calculate_rmse(prediction_topk_no_bias_cosine, test)

rmse_basic_wpearson = calculate_rmse(prediction_basic_wpearson, test)
rmse_topk_wpearson = calculate_rmse(prediction_topk_wpearson, test)
rmse_no_bias_wpearson = calculate_rmse(prediction_no_bias_wpearson, test)
rmse_topk_no_bias_wpearson = calculate_rmse(prediction_topk_no_bias_wpearson, test)
#-----------------------------------------------------------------------------#

#----------------------------------Output-------------------------------------#
print('Basic (Pearson) RMSE: ', rmse_basic_pearson)
print('Top-k (Pearson) RMSE: ', rmse_topk_pearson)
print('No Bias (Pearson) RMSE: ', rmse_no_bias_pearson)
print('Top-k & No Bias (Pearson) RMSE:', rmse_topk_no_bias_pearson, '\n')

print('Basic (Cosine) RMSE: ', rmse_basic_cosine)
print('Top-k (Cosine) RMSE: ', rmse_topk_cosine)
print('No Bias (Cosine) RMSE: ', rmse_no_bias_cosine)
print('Top-k & No Bias (Cosine) RMSE: ', rmse_topk_no_bias_cosine, '\n')

print('Basic (Weighted Pearson) RMSE: ', rmse_basic_wpearson)
print('Top-k (Weighted Pearson) RMSE: ', rmse_topk_wpearson)
print('No Bias (Weighted Pearson) RMSE: ', rmse_no_bias_wpearson)
print('Top-k & No Bias (Weighted Pearson) RMSE :', rmse_topk_no_bias_wpearson)
#-----------------------------------------------------------------------------#

