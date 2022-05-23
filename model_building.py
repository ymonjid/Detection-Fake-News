#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:28:05 2022

@author: ymonjid
"""

# I) Importing required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from models import TFIDFModel, TFIDF_KNN, NgramModel

# II) Testing the models
def testing_models(data):
    df_models = pd.DataFrame(columns=('model', 'score'))
    # 1) Building TFIDF models
    SupportVectorClassifier=svm.SVC(kernel='linear')
    print('Models with Tfidf Feature extraction Techniques : \n')
    print('************************************************ \n')
    acc_TFIDF_LogReg = TFIDFModel(data, Model=LogisticRegression(),txt='Logistic Regression Model : \n ')
    i=0; df_models.loc[i] = ['TFIDF_LogReg', acc_TFIDF_LogReg]
    acc_TFIDF_svm = TFIDFModel(data, Model=SupportVectorClassifier,txt='Support Vector Classifier Model : \n ')
    i = i+1; df_models.loc[i] = ['TFIDF_svm', acc_TFIDF_svm]
    acc_TFIDF_DecTree = TFIDFModel(data, Model=tree.DecisionTreeClassifier(),txt='Decision Tree Classifier Model : \n ')
    i = i+1; df_models.loc[i] = ['TFIDF_DecTree', acc_TFIDF_DecTree]
    # Very low accuracy with KNN
    #acc_TFIDF_KNN=KNN_TFIDF(data)
    #i = i+1
    #df_models.loc[i] = ['TFIDF_KNN', acc_TFIDF_KNN]
    
    # 2) Building Ngram models
    print('Models with Ngram Feature extraction Techniques : \n')
    print('************************************************ \n')
    acc_Ngram_LogReg2=NgramModel(data, Model=LogisticRegression(),txt='Logistic Regression Model : \n ', n=2)
    i = i+1; df_models.loc[i] = ['Ngram_LogReg_2', acc_Ngram_LogReg2]
    acc_Ngram_LogReg3=NgramModel(data, Model=LogisticRegression(),txt='Logistic Regression Model : \n ', n=3)
    i = i+1; df_models.loc[i] = ['Ngram_LogReg_3', acc_Ngram_LogReg3]

    acc_Ngram_svm2=NgramModel(data, Model=SupportVectorClassifier ,txt='Support Vectoer Classifier Model : \n ', n=2)
    i = i+1; df_models.loc[i] = ['Ngram_svm_2', acc_Ngram_svm2]
    acc_Ngram_svm3=NgramModel(data, Model=SupportVectorClassifier ,txt='Support Vectoer Classifier Model : \n ', n=3)
    i = i+1; df_models.loc[i] = ['Ngram_svm_3', acc_Ngram_svm3]

    acc_Ngram_DecTree2=NgramModel(data, Model=tree.DecisionTreeClassifier(),txt='Decision Tree Classifier Model : \n ', n=2)
    i = i+1; df_models.loc[i] = ['Ngram_DecTree_2', acc_Ngram_DecTree2]
    acc_Ngram_DecTree3=NgramModel(data, Model=tree.DecisionTreeClassifier(),txt='Decision Tree Classifier Model : \n ', n=3)
    i = i+1; df_models.loc[i] = ['Ngram_DecTree_3', acc_Ngram_DecTree3]

    return df_models
