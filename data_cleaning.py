#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:54:34 2022

@author: ymonjid
"""
# Importing the required libraries for text cleaning
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()
wnl = nltk.stem.WordNetLemmatizer()
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

def clean(text):
    text = str(text).replace(r'http[\w:/\.]+', ' ')  # removing urls
    text = str(text).replace(r'[^\.\w\s]', ' ')  # remove everything but characters and punctuation
    text = str(text).replace('[^a-zA-Z]', ' ')
    text = str(text).replace(r'\s\s+', ' ')
    text = text.lower().strip()
    return text

def nltk_preprocess(text):
    text = clean(text)
    wordlist = re.sub(r'[^\w\s]', '', text).split()
    text = ' '.join([wnl.lemmatize(word) for word in wordlist if word not in stopwords_dict])
    return  text

def data_cleaning(data):    
    # 1) Dealing with NaN values in the title column
    data_na = data[data['title'].isna() == True]
    # a) Creating a DataFrame with the first 20 words of the text feature
    data_text_20 = []
    data_title_20 = []
    j = 0
    for i in data_na.index:
        data_text_20.append(data_na.loc[i]['text'].split(' ')[:20])
        data_title_20.append(' '.join(data_text_20[j]) )
        j = j+1
    # b) Replacing the NaN values by the first 20 words in text
    data_na.loc[data_na.index, 'title'] = data_title_20
    data.loc[data['title'].isna(), 'title'] = data_title_20
    
    # 2) Droping rows with NaN values in the text and author column (simultaneously)
    data = data.drop(data[data['text'].isna()].index)
    
    # 3) Replace the NaN values in the author column by 'Unknown author'
    data.loc[data['author'].isna(), 'author'] = 'Unknown author'
    
    # 4) Cleaning text from unused characters and Nltk Preprocessing including:
        # Stop words, Stemming and Lemmetization
    data['text'] = data.text.apply(nltk_preprocess)
    data['title'] = data.title.apply(nltk_preprocess)
    
    # 5) Adding a column with number of words in text
    for i in data.index:
        data.loc[i, 'length text'] = len(data.text[i].split(' '))
    
    return data
    

    
    