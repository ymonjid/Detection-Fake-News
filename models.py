#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:48:05 2022

@author: ymonjid
"""

# I) Importing required libraries
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# II) Defining the models
# 1) TFIDFModel: TF-IDF is an abbreviation for Term Frequency Inverse Document Frequency. 
# It is a common algorithm to transform text into a meaningful representation of numbers 
#which is used to fit machine algorithm for prediction. 
# (https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a)
def TFIDFModel(data, Model, txt):   
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=50)
    #x_train, x_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.2, random_state=50)
    
    vect      = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)
    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    model     = Model
    t0        = time.time()
    model.fit(train_vect, y_train)
    t1        = time.time()
    predicted = model.predict(test_vect)
    t2        = time.time()
    time_train= t1-t0
    time_pred = t2-t1
    
    accuracy  = model.score(train_vect, y_train)
    predicted = model.predict(test_vect)
    
    report = classification_report(y_test, predicted, output_dict=True)
    acc_score = accuracy_score(y_test, predicted)
    
    print(txt)
    print("Training time: %fs; Prediction time: %fs \n" % (time_train, time_pred))
    print('Accuracy score train set :', accuracy)
    print('Accuracy score test set  :', acc_score,'\n')
    print('Positive: ', report['1'])
    print('Negative : ', report['0'])
    print('\n -------------------------------------------------------------------------------------- \n')
    return acc_score

# 2) TFIDF with K Neighbors
def TFIDF_KNN(data):  
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=50)   
    vect = TfidfVectorizer(min_df = 5, max_df =0.8, sublinear_tf = True, use_idf = True)
    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    for k in [1,3,5,7,10]:
        model = KNeighborsClassifier(n_neighbors=k,algorithm='brute')
        t0        = time.time()
        model.fit(train_vect, y_train)
        t1        = time.time()
        predicted = model.predict(test_vect)
        t2        = time.time()
        time_train= t1-t0
        time_pred = t2-t1
        accuracy  = model.score(train_vect, y_train)
        predicted = model.predict(test_vect)
        acc_score = accuracy_score(y_test, predicted)
        print("Classification Report for k = {} is:\n".format(k))
        print("Training time: %fs ; Prediction time: %fs \n" % (time_train, time_pred))
        print('Accuracy score train set :', accuracy)
        print('Accuracy score test set  :', acc_score,'\n')
        print('\n -------------------------------------------------------------------------------------- \n')
        return acc_score
    
# 3) Ngram models
def NgramModel(data, Model , txt, n):
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=50)
    
    vect      = CountVectorizer(max_features=1000 , ngram_range=(n,n))
    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    model     = Model
    t0        = time.time()
    model.fit(train_vect, y_train)
    t1        = time.time()
    predicted = model.predict(test_vect)
    t2        = time.time()
    time_train= t1-t0
    time_pred = t2-t1
    accuracy  = model.score(train_vect, y_train)
    predicted = model.predict(test_vect)
    acc_score = accuracy_score(y_test, predicted)
    print("Models with " , n , "-grams :\n")
    print('********************** \n')
    print(txt)
    print("Training time: %fs; Prediction time: %fs \n" % (time_train, time_pred))
    print('Accuracy score train set :', accuracy)
    print('Accuracy score test set  :', acc_score,'\n')
    print('\n --------------------------------------------------------------------------------------------------- \n')
    return acc_score
    