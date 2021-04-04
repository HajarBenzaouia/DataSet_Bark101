# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:56:08 2020

@author: Hajar
"""

import numpy as np 
import pandas as pd
import re
import nltk 
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
import matplotlib.pyplot as plt
data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)
print(airline_tweets.head())
plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0]) 
print(plot_size[1])

airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')
features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = nltk.word_tokenize(processed_feature)
    stemmer = nltk.PorterStemmer()
    processed_feature_ws = [stemmer.stem(word.lower()) for word in processed_feature if not word in STOPWORDS]

    processed_features.append(processed_feature_ws)
 

from gensim.models import Word2Vec
size = 1000
window = 3
min_count = 1
workers = 3
sg = 1


word2vec_model_file = 'word2vec_' + str(size) + '.model'
stemmed_tokens = pd.Series(processed_features).values
# Train the Word2Vec Model
w2v_model = Word2Vec(stemmed_tokens, min_count = min_count, size = size, workers = workers, window = window, sg = sg)

features_final = []

for line in processed_features:
  line_avg = (np.mean([w2v_model[word] for word in line], axis=0)).tolist()
  sum_ = line_avg
  features_final.append(line_avg)
  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_final, labels, test_size=0.2, random_state=0)

print(labels.shape)
print(len(features_final))
print(sum_)

from sklearn.neighbors import KNeighborsClassifier 

KN = KNeighborsClassifier(n_neighbors=8)
clf = KN.fit(X_train,y_train)
predictionKN = KN.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test,predictionKN))
print(accuracy_score(y_test, predictionKN))