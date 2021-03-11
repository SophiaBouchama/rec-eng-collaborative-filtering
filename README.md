### The Effects of Data Sparsity in Collaborative Filtering Recommendation Systems (Experimental Study)

Please see the full paper at: /rec-eng-collaborative-filtering/Study/collaborative-filtering-paper.zip

### Intro

This project will be dealing with music-based recommendations, using a dataset compromising of a set of users, songs and listen count. In this context, recommendations can enhance user experience by assisting a user’s decision in selecting songs, rather than searching through a large collection. It can also enable them to discover new or alternative music that they may enjoy.

In this research project, a Python program has been written that explores various recommendation system implementations and how they perform with more or less data. This project uses the Million Song Data Set which contains data on users and their song listening activity in order to build a recommendation engine for users. The data set can be found here: http://millionsongdataset.com

### Problem Definition

A Collaborative Filtering recommendation system can be applied to situations where there are many users, many items, and some number of interactions between those users and items, known as ratings. Ratings can be either an explicit or implicit expression of a user’s preference towards the items in a set. In the majority of cases, there is a lack of user-item interaction because there are many items present in the domain, but users will have only interacted with very few. The task of a recommendation system is to somehow predict these ratings and therefore recommend relevant items to the target user. This becomes especially challenging when very little data is present causing the performance of algorithms to decline.

Collaborative Filtering predictions raise two important questions that will be explored to form the basis of this study. Firstly, how accurate can Collaborative Filtering algorithms be in predicting user ratings for a set of items? Secondly, how do these algorithms perform when faced with more or less data?

Using a chosen dataset, Collaborative Filtering models will be implemented and evaluated. Each algorithm will attempt to find similarities between users and items in order to compute user preferences to base the rating predictions. The performance of the algorithms will be evaluated by comparing predicted ratings to actual ratings. This will then allow conclusions to be drawn on how accurately different algorithms produce rating predictions. Experiments will then be conducted with more data and less data to observe the effect of different levels of sparsity on the performance of such algorithms.


### Libraries and tools

Python 3 (Anaconda Environment)

libraries:
- surprise (based on scikit-learn)
- pandas 
- numpy
- recmetrics
- sklearn

### Files and Directories

- **Data**:
contains the data used in this project:

  - triplets.txt - (data containing userID, songID and listen count)
  - song_metadata.csv - (song metadata: song_id, title, release, artist_name, year_of_release)

- **Models**:
 
  - memory_based_CF.py - contains code for the memory based collaborative filtering models 
  - model_based_CF.py - contains code for the memory based collaborative filtering models

- **Study**

  - collaborative-filtering-paper.zip - In depth full paper of the study 

  Contents of paper:
  1) Introduction
  2) Literature Review
  3) Requirements and Analysis
  4) Design and Implementation
  5) Evaluation
  6) Results and Discussion
  7) Conclusions

### Assumptions

Assumptions must be made in order to establish a consistent and valid recommendation engine. In this scenario, we can define the songs as the items and the listen count as the rating.

1) Since the data is Unary, the listen count is the implicit metric used to represents a user’s rating. For a given song, the higher the listen count, the more the user has preference towards the song.
2)  The Unary rating scale has a lower bound of 1 and no upper bound.
3)  If a listen count is not present for a given song, it will be represented by a rating of 0 which assumes that the song has not yet been observed.
4)  The similarity between users or items is dictated only by rating scores present in the dataset.

### Data Distribution (Long Tail Plot)

The distribution of the data takes the form of the long tail plot. From the graph below, out of the 5669 songs present, there are 1221 items present in the head and 4448 in the tail. From this, we can see that most of the data is sparse.

The data sparsity for this dataset is 0.49% meaning that on average each user has rated approximately 0.49 items.

<img width="891" alt="image" src="https://user-images.githubusercontent.com/23408575/110372950-d6846900-8046-11eb-9d42-050f208d921f.png">

### Building the Memory Based Models
<img width="144" alt="image" src="https://user-images.githubusercontent.com/23408575/110631324-4f4a0900-819e-11eb-87b1-c2125079404c.png">

#### Data Sparsity Experimentation:

An evaluation will be performed on how different levels of sparsity affect the performance of these algorithms. To carry out these experiments, two subsets of the data will be taken. One with a high sparsity and one with a lower sparsity.

In order to select two subsets of data with different levels of sparsity, it can be observed that there are some users who are more active than others. In the pre-processing stage, the data will be ordered by user activity, from most active users to the least active users. The top 25% of this data frame will be selected forming a denser dataset with sparsity of approximately 6.96%. The bottom 25% of the data frame will be selected forming a sparser dataset of approximately 0.53%. For the remainder of this report, the datasets will be referred to as ‘Top 25%’ and ‘Bottom 25%’ to identify the two datasets of varying sparsity level.

#### Pre-processing procedure:
1) Read in the files ‘triplet.txt’ and ‘song_metadata.csv’ as a data frame using pandas.
2) Merge the two data frames together so that one data frame remains capturing all the information.
3) Sort the data frame in order of user activity by counting the number of user rating occurrences.
4) Select the top 25% of the data frame and the bottom 25% of the data frame.

#### Computing Similarty:

Memory Based Collaborative Filtering algorithms have two types: User Based and Item Based. They use the ratings matrix to find user-user and item-item relationships that determine the similarity between the objects.

A user-item ratings matrix has been created from the data frame which is then used to compute the similarity between users and items by comparing the rows and columns respectively. Traditionally, Collaborative Filtering implements cosine similarity or Pearson correlation measures. In this project, both metrics will be used as part of the experimentation to determine how the metrics perform on the given data. A Pearson correlation function called ‘pearson_sim’ and a cosine similarity function called ‘cosine_sim’, have been created to compute similarity scores between object pairs. Each python function inputs two vectors at a time, then returns a similarity score between the two vectors.

Another function called ‘similarity_matrix’ has been created to generate a similarity matrix containing all the cosine or Pearson similarity scores. Either an item-item or a user-user similarity matrix will be generated, depending on whether Item Based or User Based Collaborative Filtering is being carried out.

The function inputs a given user-item matrix and a ‘type’ which can be either ‘user’ based or ‘item’ based, by default, the type will be User Based. The function iterates over rows of the user-item matrix and calls a similarity metric function to obtain similarity scores for all the rows permutations. The user-item matrix is transposed to compare the columns for Item Based approaches.

#### Computing Memory Based Predictions:

4 prediction functions have been written called ‘basic’, ‘topk’, ‘no_bias’ and ‘topk_no_bias’. They all have the following inputs as parameters: the appropriate similarity matrix, the type (‘user’ based or ‘item’ based) and the user-item matrix as a parameter.

##### 1) 'Basic' Predition (Baseline model):

A weighted score is applied based upon the similarities computed, this effectively considers all the users and how similar they are to the target user in order to estimate a prediction based on other user ratings. Similarity and weighting are proportional since the more similar the user, the more their rating should be considered. Therefore, each rating from the matrix is multiplied by the similarity score between the target user and all other users. The sum of the weighted scores is then computed resulting in a rating prediction.

This then must be normalised by the number of ratings given by user 𝑢 resulting in a ‘Basic’ prediction function which is applied to every single entry in the user-item matrix.

Likewise, for Item Based Collaborative Filtering, the similarity between items is considered instead of the user similarity. Therefore, depending on the type parameter, the same function can then be adapted for User Based and Item Based models by interchanging the similarity matrix.

##### 2) 'Top-k' Predition:

To improve the performance of the ‘Basic’ prediction function, rather than considering all of the users and all of the items to base predictions on, instead, only the most similar users or items are to be considered. The ‘Basic’ prediction function is adapted to sum over only the neighbourhood of users 𝑘 with the highest similarity score to the target user. The neighbourhood is defined by selecting the top-𝑘 most similar users which is applied to the prediction function

##### 3) ‘No Bias’ Prediction:

The main issue with the ‘Basic’ approach is that user rating scale may vary. For example, a user may rate items more highly on average, e.g. by actively listening to songs. On the other hand, a user may rate items more harshly e.g. a user who may not listen to music that frequently. Therefore, the relative difference in ratings given by users must be given more importance than the absolute ratings themselves. In order to account for this problem, the data can be normalised by computing the mean centred rating. 

To adapt this function for Item Based models, the mean item rating is calculated instead and added back to the function.

##### 3) ‘No Bias and Top-𝑘’ Prediction:

The final improvement made to the prediction function, includes all the adjustments previously considered. By combining the ‘No Bias’ function with the ‘Top-𝑘’ function, this results in a more optimised neighbourhood-based function for computing predictions.

This function is now normalised to account for the grade inflation/deflation caused by varying rating scales of individuals. It also only considers the top-𝑘 most similar users/items in the neighbourhood, discarding users/items that are too dissimilar.


#### Flowchart Summary of Models Implemented:
<img width="806" alt="image" src="https://user-images.githubusercontent.com/23408575/110631842-dd25f400-819e-11eb-941d-6a85b80fa732.png">

#### Evaluation

##### Hypotheses

Project aims and the research questions proposed must be formalised into a set of hypotheses that can be tested:

1) The ‘Top 25%’ dataset with sparsity 6.96% will result in the algorithms producing more accurate predictions than the ‘Bottom 25%’ dataset with sparsity 0.53%. In other words, the sparser dataset will produce higher RMSE scores than the denser dataset.
2) Out of all the Memory Based algorithms the ‘basic’ prediction algorithm will perform the worst and have the highest RMSE score. The ‘no_bias_top_k’ prediction algorithm will perform the best and have the lowest RMSE score.
3) Out of the Memory Based and Model Based algorithms, the Model Based algorithm will perform better producing lower RMSE scores.

#### Dataset Partitioning

After the pre-processing stage, a user-item matrix is constructed for two datasets: ‘Top 25%’ and ‘Bottom 25%’. Each matrix will be randomly split into training and testing data using a 70:30 split, this ratio is often split used when training a model.

Each Collaborative Filtering model will be trained using the training data and then evaluated against the test data. The results produced can then be compared against the hypotheses to draw conclusions.

##### Handelling Sparse Datasets

A 70:30 split was concluded to be more suitable rather than using the 𝑘-fold validation method. Since most of the data is sparse, when attempting to split the data into subgroups, each fold was left too sparse with insufficient data to produce acceptable results. Many rows and columns were left with only zeros which would have produced unreliable results due to insufficient training and testing data.

Since the data being dealt with is sparse, the vast majority of the user-item entries will be zero. In practice, the lack of data can end up resulting in many errors within the code that occur when the denominator of mathematical functions evaluate to zero. To avoid dividing by zero errors, a very small arbitrary number, 𝜀 has been added to the denominators where necessary. This ensures that the denominators are not zero during runtime. 𝜀 is set to a very small constant value of 1 × 10ì"îî which will therefore have a negligible effect on the results.
