#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:52:55 2022

@author: ymonjid
"""

import pandas as pd
from data_cleaning import data_cleaning
from model_building import testing_models

# 1) Reading the train test datasets
df_train = pd.read_csv('train.csv', index_col='id')
df_test = pd.read_csv('test.csv', index_col='id')

# 2) Cleaning the data
df_train_cleaned = data_cleaning(df_train)
df_train_cleaned.to_csv('train_clean.csv')
df_test_cleaned = data_cleaning(df_train)
df_test_cleaned.to_csv('test_clean.csv')

# 3) Model testing & building
df_models = testing_models(df_train_cleaned)

# 4) Test prediction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm

X_train = df_train_cleaned.text
#X_train = df_train_cleaned.drop('label', axis=1)
y_train = df_train_cleaned['label']

#X_test = df_test_cleaned.drop('label', axis=1)
X_test = df_test_cleaned.text

"""n=2
vect = CountVectorizer(max_features=1000 , ngram_range=(n,n))
vct = vect.fit(X_train)

train_vect = vct.transform(X_train)
test_vect = vct.transform(X_test)

model = LogisticRegression()
model.fit(train_vect, y_train)
predicted = model.predict(test_vect)
predicted"""

vect      = TfidfVectorizer(min_df = 5, max_df =0.8, sublinear_tf = True, use_idf = True)
train_vect= vect.fit_transform(X_train)
test_vect = vect.transform(X_test)
SupportVectorClassifier=svm.SVC(kernel='linear')
SupportVectorClassifier.fit(train_vect, y_train)
predicted = SupportVectorClassifier.predict(test_vect)