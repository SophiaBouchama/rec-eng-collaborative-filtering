#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Sophia Bouchama

#--------------------------IMPORT LIBRARIES-----------------------------------#
import pandas
import math
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

 
#---------------------------PRE PROCESSING------------------------------------#
# Creation of the data frame
triplets_file = 'triplet.txt'
song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

# create a sorted data frame that lists the customers with their number of occurrences
count_df = pandas.DataFrame(song_df_1.user_id.value_counts())
# join the count data frame back with the original data frame
new_index = count_df.merge(song_df_1[["user_id"]], left_index=True, right_on="user_id")
# output the original data frame in the order of the new index.
sorted_df = song_df_1.reindex(new_index.index)

#Top 25 percent and bottom 25 percent of users
n = 25
top_25_percent_users = sorted_df.head(int(len(sorted_df)*(n/100)))
bottom_25_percent_users = sorted_df.tail(int(len(sorted_df)*(n/100)))

top_25_percent_users.columns = ['user_id', 'song_id', 'listen_count']
bottom_25_percent_users.columns = ['user_id', 'song_id', 'listen_count']
#-----------------------------------------------------------------------------#

# A needs to be defined with the rating_scale
reader = Reader(rating_scale=(1, math.inf))

# The columns must correspond to user id, item id and ratings.
data_top = Dataset.load_from_df(top_25_percent_users[['user_id', 'song_id', 'listen_count']], reader)


#Split into training and testing data
train, test = train_test_split(data_top, test_size=0.3)

#Define algorithm as SVD
algo = SVD()

#Train the algorithm on the training data then test against testing data
algo.fit(train)
predictions = algo.test(test)

#Calculate RMSE
accuracy.rmse(predictions)

