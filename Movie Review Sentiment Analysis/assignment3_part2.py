import pandas as pd
import numpy as np
import re
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.feature_extraction.text as text
from sklearn.model_selection import train_test_split

df = pd.read_csv('IMDB_Dataset.csv')

df['sentiment_numeric'] = df['sentiment'].apply(lambda x: 1 if x.lower() == 'positive' else 0)

vectorizer = text.CountVectorizer(min_df=500)
X = vectorizer.fit_transform(df['review'])

df_bag_of_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df_bag_of_words.insert(0, 'sentiment', df['sentiment_numeric'])

data = df_bag_of_words.sample(frac=1).to_numpy()
scaler = StandardScaler()
x = scaler.fit_transform(data[:, 1:])

test_size = int(data.shape[0] / 10)
x_train = x[2 * test_size:]
x_validation = x[test_size:2 * test_size]
x_test = x[:test_size]

y_train = data[2 * test_size:, 0]
y_validation = data[test_size:2 * test_size, 0]
y_test = data[:test_size, 0]

hyperparameters = [0.1, 1.0, 10.0]
hyperparmeter_results = []

for value in hyperparameters:
    clf = LogisticRegression(max_iter=500, C=value)
    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_validation)

    accuracy = accuracy_score(y_validation, y_prediction)
    precision = precision_score(y_validation, y_prediction, average='binary', pos_label=1)
    recall = recall_score(y_validation, y_prediction, average='binary', pos_label=1)

    hyperparmeter_results.append({
        'C value': value,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    })


best_hyperparameter = max(hyperparmeter_results, key=lambda x: x['Accuracy'])


clf = LogisticRegression(max_iter=500, C=best_hyperparameter['C value'])
clf.fit(x_train, y_train)

y_prediction_validation = clf.predict(x_validation)

accuracy_validation = accuracy_score(y_validation, y_prediction_validation)
precision_validation = precision_score(y_validation, y_prediction_validation, average='binary', pos_label=1)
recall_validation = recall_score(y_validation, y_prediction_validation, average='binary', pos_label=1)


y_prediction_test = clf.predict(x_test)

accuracy_test = accuracy_score(y_test, y_prediction_test)
precision_test = precision_score(y_test, y_prediction_test, average='binary', pos_label=1)
recall_test = recall_score(y_test, y_prediction_test, average='binary', pos_label=1)