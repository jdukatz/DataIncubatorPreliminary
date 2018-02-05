import sys
from collections import Counter

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def train_classifier(data, vectors, classifier):
    """
    given data and a vectorization of that data, trains the
    provided classifier using 10-fold cross validation
    then returns the trained classifier
    """

    skf = StratifiedKFold(train_labels, n_folds=10, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(skf):
        print('results for fold {}'.format(i))
        # separate train and test indices
        train_points = vectors[train_idx]
        train_y = train_labels[train_idx]

        test_points = vectors[test_idx]
        test_y = train_labels[test_idx]

        # fit the classifier then make a prediction on the test fold
        classifier.fit(train_points, train_y)
        predictions = classifier.predict(test_points)
        report = classification_report(
            y_true=test_y,
            y_pred=predictions,
            labels=le.transform(le.classes_),
            target_names=le.classes_)
        print(report)

    return classifier

def classify(trained_classifier, test_point):
    """Classify a single point given a trained classifier"""
    print('true:')
    print(test_point['channel'], test_point['text'])
    print('prediction:')
    print(le.inverse_transform(nb.predict(tfidf.transform(test_chyron['text']))))


def cluster(vectors, k):
    """Use sklearn's implementation to generate k clusters"""
    clusters = KMeans(n_clusters=k).fit(vectors)
    return clusters

# read in data, generate test and train sets
in_file = sys.argv[1]
data = pd.read_csv(in_file, sep='\t')
train_data, test_data = train_test_split(data, test_size=0.2)

# encode the labels
le = LabelEncoder()
le.fit(data['channel'].values)
train_labels = le.transform(train_data['channel'].values)
test_labels = le.transform(test_data['channel'].values)

# convert the data to tfidf matrices
tfidf = TfidfVectorizer()
tfidf.fit(data['text']) # fit to all the data in case some words only appear in testing or testing
train_vectors = tfidf.transform(train_data['text'])
test_vectors = tfidf.transform(test_data['text'])

# cluster all data
all_vectors = tfidf.transform(data['text'])
km_clusters = cluster(all_vectors, 4)

# gather some info about the words used in each cluster
for cluster in range(4):
    words = Counter()
    for row in range(len(data['text'].values)):
        if km_clusters.labels_[row] == cluster:
            text = word_tokenize(data.iloc[row]['text'])
            words.update([token for token in text if token.lower() not in stopwords.words('english')])
    print('10 most common words in cluster {}: {}'.format(cluster, words.most_common(10)))

# Naive bayes
print('Training naive bayes model')
nb = MultinomialNB()
trained_nb = train_classifier(train_data, train_vectors, nb)
print('Testing on held out set')
# make a prediction with the trained classifier on held out data
test_preds = trained_nb.predict(test_vectors)
nb_report = classification_report(
    y_true=test_labels,
    y_pred=test_preds,
    labels=le.transform(le.classes_),
    target_names=le.classes_)
# generate confusion matrix
nb_cnf_matrix = confusion_matrix(test_labels, test_preds)
plt.figure()
plt.imshow(nb_cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Naive Bayes predictions')
plt.colorbar()
tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_)
plt.yticks(tick_marks, le.classes_)
plt.ylabel('True Network')
plt.xlabel('Predicted Network')
plt.show()
# print report
print(nb_report)

# svm
print('Training svm')
svm = SVC(kernel='linear')
trained_svm = train_classifier(train_data, train_vectors, svm)
print('Testing on held out set')
# make a prediction with the trained classifier on held out data
test_preds = trained_svm.predict(test_vectors)
svm_report = classification_report(
    y_true=test_labels,
    y_pred=test_preds,
    labels=le.transform(le.classes_),
    target_names=le.classes_)

# generate confusion matrix
svm_cnf_matrix = confusion_matrix(test_labels, test_preds)
plt.figure()
plt.imshow(svm_cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('SVM predictions')
plt.colorbar()
tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_)
plt.yticks(tick_marks, le.classes_)
plt.ylabel('True Network')
plt.xlabel('Predicted Network')
plt.show()
# print report
print(svm_report)


