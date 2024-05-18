# Movie Review Sentiment Analysis

This repository contains the implementation of a machine learning project to classify movie reviews as positive or negative using the Large Movie Review Data Set compiled by researchers at Stanford.

## Data Description

The data set contains 50,000 movie reviews:
- 25,000 highly positive reviews
- 25,000 highly negative reviews

The data is provided in two formats:
1. **IMDB Dataset.csv**: Contains two columns, one with the original text of the review and the other with a positive or negative label.
2. **IMDB BOW.pkl**: A compressed data frame with the same data in a “bag of words” representation. The first column contains the sentiment label (1 for positive, 0 for negative), and the remaining columns correspond to words appearing in the reviews.

## Prerequisites

- Python 3.x
- Scikit-learn
- Pandas
- NumPy

## Instructions

### 1. Algorithms

Split the data into training, validation, and test sets. Perform any necessary preprocessing (e.g., scaling, randomizing order).

Select three machine learning algorithms suitable for binary classification and use them to classify the data. Report accuracy, precision, and recall on the validation set for each algorithm. Suggested algorithms include:
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forests

Identify the best performing algorithm and discuss its performance.

### 2. Hyperparameters

Focus on the best performing algorithm from part 1. Select a hyperparameter and describe its effect on the algorithm. Train the classifier using three different values for this hyperparameter. Report accuracy, precision, and recall for each value and discuss the results.

### 3. Features

Enhance the best model from part 2 by adding new features. Extract new features from `IMDB Dataset.csv` and add them to your feature vectors. Suggested new features include:
- Count of certain punctuation marks
- Count of words in all caps
- Frequency of short multi-word phrases
- Count of positive and negative words using an opinion lexicon

Describe the new features and train the classifier with the updated feature vectors. Report validation accuracy, precision, and recall, and compare the results with the previous model.

### Final Model

Use the best model to classify the test set. Report test accuracy, precision, and recall. Discuss the performance and suggest improvements.

## Files

- `IMDB Dataset.csv`: Original movie reviews with sentiment labels.
- `IMDB BOW.pkl`: Bag of words representation of the reviews.
- `multi vocab`: List of words used as features.
- `get bow rep.py`: Script used to convert data into the bag of words representation.
- `assignment2.py`: Sample script for data preprocessing and training a logistic regression classifier.


## Results

Summarize the results of your experiments, including the performance of different algorithms, the impact of hyperparameter tuning, and the effect of new features on the model's performance.

