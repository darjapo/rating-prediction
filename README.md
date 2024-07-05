# Movie rating prediction system

This project was created in scope of Big Data Analytics course in Sirindhorn International Institute of Technology. The aim of the project is to created a movie prediction system (predict missing ratings for movies) based on MovieLens dataset, while obtaining the minimum possible root-mean-square error (RMSE).

## Overview

For this project three datasets were provided: training dataset, validation dataset and movie dataset. Both training and validation datasets contain following columns: userId, movieId and rating. Movie dataset contains movieId, title and genres. Threee recommendation system approaches were tested and compaired: using Cosine similarity matrix, Person similarity matrix and latent factor model with biases. Both Cosine and Pearson similarity approaches are user-based ones. Only training dataset was used for rating prediction. The goal stated in the project's description was to obtain an RMSE less than 0.9171.

## Approaches
<details>
  
  <summary>
    
### 1. Cosine similarity
  </summary>
  
  - Create Utility Matrix:
    - Build a matrix from the training data with users as rows, movies as columns, and ratings as values. Fill in missing ratings with zeros.
  - Compute Cosine Similarity:
    - Calculate a similarity matrix to measure how similar users are to each other based on their ratings.
  - Predict Ratings:
    - For each user-movie pair in the validation set, predict ratings using the weighted average ratings from similar users.
    - The formula used to make predictions in this movie recommendation system is based on the weighted ratings from similar users. In simple terms, movie rating left by similar users is multiplied with the similarity value between the target user and the corresponging similar user. These values are summed and divided by the sum of similarity values.
    - If the sum of the similarity scores is zero, the prediction defaults to the average rating for that movie. If there are no ratings for the movie at all, it defaults to zero.
  - Evaluate Performance:
    - Compute the Root Mean Squared Error (RMSE) to compare the predicted ratings with the actual ratings. Result: **RMSE = 0.9961**.
</details>

<details>
  
  <summary>

### 2. Pearson similarity
  </summary>

  - Create Utility Matrix:
    - Build a matrix from the training data with users as rows, movies as columns, and ratings as values. Fill in missing ratings with zeros.
  - Compute Pearson Similarity:
    - Calculate a similarity matrix to measure how similar users are to each other based on their ratings. This involves centering the ratings by subtracting the mean rating for each movie and then computing the Pearson correlation coefficient between users.
  - Predict Ratings:
    - For each user-movie pair in the validation set, predict ratings using the weighted average ratings from similar users. Adjust the ratings by the users' average ratings to improve accuracy.
    - The formula used to make predictions in this movie recommendation system is based on calculating the predicted rating by adjusting the average rating of the target user using deviations from the average ratings of similar users, weighted by their similarity scores. In simple terms, movie rating left by similar users is multiplied with the similarity value between the target user and the corresponging similar user. These values are summed and divided by the sum of similarity values.
    - If there are no similar users who have rated movie, the prediction defaults to the average rating of user. If there are no ratings for movie at all, it defaults to zero.
  - Evaluate Performance:
    - Compute the Root Mean Squared Error (RMSE) to compare the predicted ratings with the actual ratings. Result: **RMSE = 0.8901**.
</details>

<details>

  <summary>

### 3. Latent factor model with biases (Matrix Factorization with Stochastic Gradient Descent (SGD))
  </summary>
  
  - Data Preparation:
    - Create the user-item matrix, where rows represent users, columns represent movies, and values represent ratings.
    - Perform Singular Value Decomposition (SVD) on this matrix to initialize user and movie factor matrices with the specified number of latent factors.
  - Bias Calculations:
    - Global Bias: Average rating of all users for all movies.
    - User Biases: Average rating of each user minus the global bias.
    - Item Biases: Average rating of each movie minus the global bias.
  - Model Training:
    - Update user and item biases and factor matrices using the gradient descent method.
    - For each user-item pair in the training data:
      - Predict the rating.
      - Compute the error between the actual and predicted rating.
      - Adjust biases and factors to minimize the error, considering the regularization term.
  - Rating Prediction:
    - Use the trained biases and factor matrices to predict ratings for the user-item pairs in the validation dataset.
    - Calculate the predicted rating by combining the global bias, user bias, item bias, and dot product of user and item factors. The formula is based on the sum of global bias, updated user and item biases, and a dot product of updated user and item factor vectors of target user and movie.
  - Evaluate Performance:
    - Compute the Root Mean Squared Error (RMSE) to compare the predicted ratings with the actual ratings. Result: **RMSE = 0.8340**.
</details>

## Analysis

The best resulted based on the RMSE value was achieved through the latent factor model with biases approah. However, it is important to note that the number of latent factors was not calculated, all the constants for the model training were chosen based on the code runs and comparison of gotten results. Possible way of making the error lower would be to calculate the latent factor number for given dataset. Another improvement could be to combine approaches in ensemble recommendation system.
