### The Effects of Data Sparsity in Collaborative Filtering Recommendation Systems (Experimental Study)

Please see the full paper at: /rec-eng-collaborative-filtering/collaborative-filtering-paper.zip

### Intro
This project uses the Million Song Data Set which contains data on users and their song listening activity in order to build a recommendation engine for users. The data set can be found here: http://millionsongdataset.com

### Problem Definition
Before the digital age, relying on the opinions and tastes of a trusted cohort helped us to make decisions involving which products we chose to buy or what music we may enjoy listening to. Today the wealth of information available to us online brings forth the problem of information overload. As a result, recommendation systems have been developed to tackle this by allowing users to automatically filter through vast amounts of content.

The task of a recommendation system is to attempt to compute accurate and relevant recommendations to users. In most practical settings, users will have insufficient experience with the majority of items available to them, therefore the challenge lies in creating accurate recommendations when a lack of data is present. In this research project, a Python program has been written that explores various recommendation system implementations and how they perform with more or less data.

A Collaborative Filtering recommendation system can be applied to situations where there are many users, many items, and some number of interactions between those users and items, known as ratings. Ratings can be either an explicit or implicit expression of a user’s preference towards the items in a set. In the majority of cases, there is a lack of user-item interaction because there are many items present in the domain, but users will have only interacted with very few. The task of a recommendation system is to somehow predict these ratings and therefore recommend relevant items to the target user. This becomes especially challenging when very little data is present causing the performance of algorithms to decline.

Collaborative Filtering predictions raise two important questions that will be explored to form the basis of this study. Firstly, how accurate can Collaborative Filtering algorithms be in predicting user ratings for a set of items? Secondly, how do these algorithms perform when faced with more or less data?

Using a chosen dataset, Collaborative Filtering models will be implemented and evaluated. Each algorithm will attempt to find similarities between users and items in order to compute user preferences to base the rating predictions. The performance of the algorithms will be evaluated by comparing predicted ratings to actual ratings. This will then allow conclusions to be drawn on how accurately different algorithms produce rating predictions. Experiments will then be conducted with more data and less data to observe the effect of different levels of sparsity on the performance of such algorithms.

There are many contexts in which recommendations could benefit a user, this project will be dealing with music-based recommendations, using a dataset compromising of a set of users, songs and listen count. In this context, recommendations can enhance user experience by assisting a user’s decision in selecting songs, rather than searching through a large collection. It can also enable them to discover new or alternative music that they may enjoy.

### Libraries and tools

Python 3 (Anaconda Environment)

libraries:
- surprise (based on scikit)
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
4)  The similarity between users or items is dictated only by rating scores present in the
dataset.

### Data Distribution (Long Tail Plot)

<img width="891" alt="image" src="https://user-images.githubusercontent.com/23408575/110372950-d6846900-8046-11eb-9d42-050f208d921f.png">


