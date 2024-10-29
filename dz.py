import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = pd.read_csv('IMDB Dataset.csv')
texts = data['review']
labels = data['sentiment']

texts = texts.str.lower().str.replace('[^\w\s]', '', regex=True)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

svd = TruncatedSVD(n_components=100)
word_vectors = svd.fit_transform(tfidf_matrix)

def get_synonyms(word, n=5):
    if word not in vectorizer.get_feature_names_out():
        return []

    word_index = vectorizer.vocabulary_[word]

    if word_index >= len(word_vectors):
        return []

    word_vector = word_vectors[word_index].reshape(1, -1)
    similarities = cosine_similarity(word_vector, word_vectors)

    similar_indices = np.argsort(similarities[0])[::-1][1:n + 1]
    unique_synonyms = set(vectorizer.get_feature_names_out()[i] for i in similar_indices)

    return list(unique_synonyms)


def get_antonyms(word, n=5):
    synonyms = get_synonyms(word, n)
    antonyms = []

    for synonym in synonyms:
        synonym_antonyms = get_synonyms(synonym, n=1)
        antonyms.extend(synonym_antonyms)

    return list(set(antonyms))


kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(tfidf_matrix, labels):
    X_train, X_test = tfidf_matrix[train_index], tfidf_matrix[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

    clf = LogisticRegression(C=1.0, solver='liblinear')
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')

    print(classification_report(y_test, predictions, target_names=['negative', 'positive']))

word = "good"
print(f"Синоніми для '{word}': {get_synonyms(word)}")
print(f"Антоніми для '{word}': {get_antonyms(word)}")

"""

https://drive.google.com/file/d/1TeqyXzPK8OQ09ljgEw0A-UoUo1iJih8B/view?usp=drive_link

Accuracy: 0.8978
              precision    recall  f1-score   support

    negative       0.91      0.88      0.90      5000
    positive       0.89      0.91      0.90      5000

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000

Accuracy: 0.8934
              precision    recall  f1-score   support

    negative       0.90      0.88      0.89      5000
    positive       0.89      0.90      0.89      5000

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

Accuracy: 0.8922
              precision    recall  f1-score   support

    negative       0.90      0.88      0.89      5000
    positive       0.89      0.90      0.89      5000

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

Accuracy: 0.8951
              precision    recall  f1-score   support

    negative       0.91      0.88      0.89      5000
    positive       0.89      0.91      0.90      5000

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000

Accuracy: 0.898
              precision    recall  f1-score   support

    negative       0.90      0.89      0.90      5000
    positive       0.90      0.90      0.90      5000

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000

Синоніми для 'good': []
Антоніми для 'good': []

Process finished with exit code 0

"""
