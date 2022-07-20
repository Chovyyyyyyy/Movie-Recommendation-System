# Movie-Recommendation-System

## Dataset used

1. [The Movie Database](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

## Aim

Build a movie recommendation system by integrating the aspects of personalization of user with the overall features of movie such as genre, popularity etc.

## Models

* Popularity model
* Content based model: genre, year of release, ratings of movies
* Collaborative filtering: User vs item, KNN similarity measures
* Latent Factor based SVD
* Combined linear model using surprise library (CF + SVD)
* Hybrid model (content based + popularity based + item-item CF + svd)

## 1_code Folder Description

### 1. preprocessing

Code for spliting the data into training and testing set for each user such that 80% ratings are in training and 20% are for testing.

### 2. simple_recommender

A simple recommender.

### 3. content_based

Generate user and movie vectors based on genre and predicting the ratings for movies in test data.

### 3. content_based@movie_similarity

Content based approach to include the user's genre preference and recommend movies similar to user's highly rated movies.

### 4. collab_filtering@basic_surprise

Code for generating ratings for test data using surprise models such as KNN (CF), SVD, and quality of recommendations (precision, recall, ndcg, f-measure).

### 4. collab_filtering@knn_analysis

Analysis of KNN algorithms by changing different parameters like:

* number of neighbors
* similarity metrices
* user v/s item based CF

### 4. collab_filtering@hyperparameter_tuning

Fine-tuned surprise models by experimenting with different hyperparameters for training and model. Compared models based on RMSE and MAE.

### 5. combined_model

Combination of different surprise model results by applying weighted linear combination to generate final rating.

## Results

![Hybrid model](Results/images/Hybrid_Model.png)

All the models are implemented in Python using pandas, sklearn and [surprise](http://surpriselib.com/) library. The hyperparameter tuning, testing accuracy (RMSE and MAE) and evaluation of recommendations (precision, recall, f-measure and ndcg) for each model are thoroughly performed. The detailed analysis of the models is presented in the report.
