# Importing Essentials
import pandas as pd
from sklearn import metrics
import re
import gensim
from gensim.sklearn_api import D2VTransformer
import numpy as np
from nltk import PorterStemmer, SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Conv1D, GlobalMaxPooling1D
from bert_sklearn import BertClassifier
from nltk.tokenize import word_tokenize
from gensim.sklearn_api import W2VTransformer

nltk.download('punkt')


def cleaning(sentence):

    ps = PorterStemmer()
    # remove hashtags
    sentence = re.sub(
        "(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", ' ', sentence)
    # remove weird chars
    sentence = re.sub('[^a-zA-z\'\"]+', ' ', sentence)
    # remove urls
    sentence = re.sub(r'\$\w*', '', sentence)
    # remove old style retweet text "RT"
    sentence = re.sub(r'^RT[\s]+', '', sentence)
    # remove hyperlinks
    sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence)
    # only removing the hash # sign from the word
    sentence = re.sub(r'#', '', sentence)

    sentence = sentence.lower()
    sentence = sentence.split()

    stemmimg_words = [ps.stem(word)
                      for word in sentence if not word in stopwords.words('english')]

    sentence = ' '.join(stemmimg_words)
    return sentence


def get_part_of_day(hour):

    if 5 <= hour <= 11:
        return 'morning'

    elif 12 <= hour <= 17:
        return "afternoon"

    elif 18 <= hour <= 22:
        return 'evening'

    else:
        return "night"


def get_is_weekend(day):

    if(day == "Thu" or day == "Fri" or day == "Sat"):
        return "weekend"
    return "not weekend"


def get_data_with_date(data):
    sen_w_feats = []

    # The labels for the samples.
    labels = []

    for index, row in data.iterrows():
        date = row["date"].split(' ')

        # Piece it together...
        combined = ""

        # combined += "The ID of this item is {:}, ".format(row["Clothing ID"])
        combined += "Now it's {:}. today is {:}, {:}th of {:}, " \
            "today is {:}. ".format(get_part_of_day(int(date[3][:2])),
                                    date[0],
                                    date[2],
                                    date[1],
                                    get_is_weekend(date[0]))
        combined += row['text']
        sen_w_feats.append(combined)

        # Also record the sample's label.
        labels.append(row['label'])
    return sen_w_feats, labels


def run_models(X, y, vect):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=0.2)

    vect.fit(X_train)
    X_train_dtm = vect.transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # Accuracy using Naive Bayes Model
    NB = MultinomialNB()
    NB.fit(X_train_dtm, y_train)
    y_pred = NB.predict(X_test_dtm)
    print('\nNaive Bayes')
    print('Accuracy Score: ', metrics.accuracy_score(
        y_test, y_pred)*100, '%', sep='')
    print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')

    # Accuracy using Logistic Regression Model
    LR = LogisticRegression()
    LR.fit(X_train_dtm, y_train)
    y_pred = LR.predict(X_test_dtm)
    print('\nLogistic Regression')
    print('Accuracy Score: ', metrics.accuracy_score(
        y_test, y_pred)*100, '%', sep='')
    print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')

    # Accuracy using SVM Model
    SVM = LinearSVC()
    SVM.fit(X_train_dtm, y_train)
    y_pred = SVM.predict(X_test_dtm)
    print('\nSupport Vector Machine')
    print('Accuracy Score: ', metrics.accuracy_score(
        y_test, y_pred)*100, '%', sep='')
    print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')

    # Accuracy using KNN Model
    KNN = KNeighborsClassifier(n_neighbors=317)
    KNN.fit(X_train_dtm, y_train)
    y_pred = KNN.predict(X_test_dtm)
    print('\nK Nearest Neighbors (NN = 3)')
    print('Accuracy Score: ', metrics.accuracy_score(
        y_test, y_pred)*100, '%', sep='')
    print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')


if __name__ == "__main__":
    col = ['label', 'ids', 'date', 'flag', 'user', 'text']
    data = pd.read_csv('dataset.csv', header=None, names=col,
                       encoding='latin-1').dropna()
    data = data[['text', 'label', 'date']]
    data['label'] = data['label'].replace(4, 1)
    data_pos = data[data['label'] == 1]
    data_neg = data[data['label'] == 0]
    # print(len(data_pos))
    # print(len(data_neg))
    data_pos = data_pos.iloc[:int(10000)]
    data_neg = data_neg.iloc[:int(10000)]
    data = pd.concat([data_pos, data_neg])
    data['text'] = data['text'].apply(lambda text: cleaning(text))
    X_only_text = data.text
    y_only_text = data.label
    print("Sentiment analyses with only text")
    vectCV = CountVectorizer(stop_words='english',
                             max_df=.80, min_df=4)
    vectTF = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    print("Bag of words:")
    run_models(X_only_text, y_only_text, vectCV)

    print("TfIdf:")
    run_models(X_only_text, y_only_text, vectTF)

    print("Sentiment analyses with additional features:")
    X_w_feats, y_w_feats = get_data_with_date(data)
    X_w_feats = map(cleaning, X_w_feats)

    print("Bag of words:")
    run_models(X_w_feats, y_w_feats, vectCV)

    print("TfIdf:")
    run_models(X_w_feats, y_w_feats, vectTF)
