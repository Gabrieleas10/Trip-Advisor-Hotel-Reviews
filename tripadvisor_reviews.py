# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor , XGBClassifier
import nltk
from nltk.tag import pos_tag
from functools import reduce
from itertools import groupby
import numpy as np
import seaborn as sns

# importing dataset
base = pd.read_csv('tripadvisor_hotel_reviews.csv' , sep=',')

# plotting value counts of Rating (Unbalaced dataset)
sns.countplot(x='Rating', data= base)

# counting max samples
max_samples = len(base[base['Rating']== 5])

# spliting dataset
df_maj = base[(base['Rating'] == 5) ]

# creating balanced datasets
df_q1 = base[base['Rating'] == 1]
df_q2 = base[base['Rating'] == 2]
df_q3 = base[base['Rating'] == 3]
df_q4 = base[base['Rating'] == 4]

dfm1 = resample(df_q1 ,
                replace = True,
                n_samples= max_samples,
                random_state=0)

dfm2 = resample(df_q2 ,
                replace = True,
                n_samples= max_samples,
                random_state=0)

dfm3 = resample(df_q3 ,
                replace = True,
                n_samples= max_samples,
                random_state=0)

dfm4 = resample(df_q4 ,
                replace = True,
                n_samples= max_samples,
                random_state=0)

# creating final dataset balaced
base = pd.concat([df_maj , dfm1 , dfm2 , dfm3 , dfm4])

# spliting review words
base['Review'] = base['Review'].apply(lambda x: x.split()) 

base['Review'] = base['Review'].apply(lambda x:[item.replace(',','') for item in x if item.count(',')!=-1])

# importing list of words: positive and negative
eas = os.path.join('C:\\Users\\GABRIEL\\Downloads\\')
pos_words_file = os.path.join(eas, 'positive_words.txt')
neg_words_file = os.path.join(eas, 'negative_words.txt')

pos_words = []
neg_words = []

# reading .txt files
for pos_word in open(pos_words_file, 'r').readlines():
    pos_words.append(pos_word.replace('\n',''))

for neg_word in open(neg_words_file, 'r').readlines():
    neg_words.append(neg_word.replace('\n',''))

# counting positive words in review
base['Positive'] = base['Review'].apply(lambda x: len((set(x) & set(pos_words))))
# counting negative words in review
base['Negative'] = base['Review'].apply(lambda x: len((set(x) & set(neg_words))))
# total of words
base['Total'] = base['Review'].apply(lambda x: len(x))
# value of neutral words
base['Neutro'] = base['Total'] - (base['Positive'] + base['Negative'])
# 
base['Positive%'] = base['Positive'] / base['Total'] 

base['Negative%'] = base['Negative'] / base['Total']

base['Delta'] = base['Positive'] - base['Negative']

base['First_Words_P'] = base['Review'].apply(lambda x: len((set(x[0:3]) & set(pos_words))))

base['First_Words_N'] = base['Review'].apply(lambda x: len((set(x[0:3]) & set(neg_words))))

base['Tags'] = base['Review'].apply(lambda x: pos_tag(x))
    
