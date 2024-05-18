
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import sklearn.feature_extraction.text as text
import pickle

df = pd.read_csv('IMDB_Dataset.csv')
df['sentiment_numeric'] = df['sentiment'].apply(lambda x: 1 if x.lower() == 'positive' else 0)

vectorizer = text.CountVectorizer(min_df=500)
X = vectorizer.fit_transform(df['review'])

df_bag_of_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df_bag_of_words.insert(0, 'sentiment', df['sentiment_numeric'])

with open('IMDB_BOW.pkl', 'wb') as f:
    pickle.dump(df_bag_of_words, f)

data = df_bag_of_words.to_numpy()
np.random.shuffle(data)

scaler = StandardScaler()
x = scaler.fit_transform(data[:, 1:])

test_size = int(data.shape[0] / 10)

x_train = x[2 * test_size:]
x_val = x[test_size:2 * test_size]
x_test = x[:test_size]

y_train = data[2 * test_size:, 0]
y_val = data[test_size:2 * test_size, 0]
y_test = data[:test_size, 0]


clf_logistic_regression = LogisticRegression(max_iter=500)
clf_logistic_regression.fit(x_train, y_train)
y_prediction_logistic_regression = clf_logistic_regression.predict(x_val)

accuracy_logistic_regression = metrics.accuracy_score(y_val, y_prediction_logistic_regression)
precision_logistic_regression = metrics.precision_score(y_val, y_prediction_logistic_regression, pos_label=1)
recall_logistic_regression = metrics.recall_score(y_val, y_prediction_logistic_regression, pos_label=1)


clf_svm = SVC()
clf_svm.fit(x_train, y_train)
y_prediction_svm = clf_svm.predict(x_val)

accuracy_svm = metrics.accuracy_score(y_val, y_prediction_svm)
precision_svm = metrics.precision_score(y_val, y_prediction_svm, pos_label=1)
recall_svm = metrics.recall_score(y_val, y_prediction_svm, pos_label=1)


clf_random_forest = RandomForestClassifier(n_estimators=100)
clf_random_forest.fit(x_train, y_train)
y_prediction_random_forest = clf_random_forest.predict(x_val)

accuracy_random_forest = metrics.accuracy_score(y_val, y_prediction_random_forest)
precision_random_forest = metrics.precision_score(y_val, y_prediction_random_forest, pos_label=1)
recall_random_forest = metrics.recall_score(y_val, y_prediction_random_forest, pos_label=1)


best_algorithm = "Logistic Regression" if accuracy_logistic_regression > accuracy_svm and accuracy_logistic_regression > accuracy_random_forest else (
    "SVM" if accuracy_svm > accuracy_random_forest else "Random Forest")

print("Best algorithm :", best_algorithm)