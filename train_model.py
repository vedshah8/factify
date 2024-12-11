import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# Load and prepare datasets
fake_data = pd.read_csv('Fake.csv')
real_data = pd.read_csv('True.csv')
real_data = real_data.drop(columns=['date'])
fake_data = fake_data.drop(columns=['date'])

# Label data
fake_data['label'] = 0
real_data['label'] = 1

# Combine datasets
data = pd.concat([fake_data, real_data], ignore_index=True)

# Split features and labels
X = data[['title', 'text']]
y = data['label']

# Vectorize and encode features
text_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
subject_encoder = OneHotEncoder(handle_unknown='ignore')

# Fit and transform training data
text_train = text_vectorizer.fit_transform(X['text'])

title_train = text_vectorizer.fit_transform(X['title'])

weighted_train = hstack([text_train * 1.0, title_train * 1.0])

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Added max_iter in case convergence issues
model.fit(weighted_train, y)

import pickle

# Save the TfidfVectorizer to a file
with open('text_vectorizer.pkl', 'wb') as file:
    pickle.dump(text_vectorizer, file)

# Save the trained model to another file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)
