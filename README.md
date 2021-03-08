### Study: The Effects of Data Sparsity in Collaborative Filtering Recommendation Systems

Please see the full paper at: /rec-eng-collaborative-filtering/collaborative-filtering-paper.zip

### Intro
This project uses the Million Song Data Set which contains data on users and their song listening activity in order to build a recommendation engine for users. The data set can be found here: http://millionsongdataset.com

### Problem Definition

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

song_metadata.csv - contains the data used in this project

triplets.txt

- **Models**:

memory_based_CF.py - contains code for the memory based collaborative filtering models (metadata containing users, songs and listen count) 

model_based_CF.py - contains code for the memory based collaborative filtering models

- **collaborative-filtering-paper.zip** - In depth full paper of the study 

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
5)The similarity between users or items is dictated only by rating scores present in the
dataset.

### Data Distribution (Long Tail Plot)

<img width="891" alt="image" src="https://user-images.githubusercontent.com/23408575/110372950-d6846900-8046-11eb-9d42-050f208d921f.png">


