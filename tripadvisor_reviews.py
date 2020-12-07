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
#from xgboost import XGBRegressor , XGBClassifier
import nltk
from nltk.tag import pos_tag
from functools import reduce
from itertools import groupby
import numpy as np
import seaborn as sns

# installing nltk packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

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
eas = os.path.join('E:\\Projects\\Trip-Advisor-Hotel-Reviews\\words_list\\')
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
# percentage of positive words of review
base['Positive%'] = base['Positive'] / base['Total'] 
# percentage of negative words of review
base['Negative%'] = base['Negative'] / base['Total']
# difference between positive and negative words of review
base['Delta'] = base['Positive'] - base['Negative']
# counting first positives words of review (0 to 3° word)
base['First_Words_P'] = base['Review'].apply(lambda x: len((set(x[0:3]) & set(pos_words))))
# counting first negatives words of review (0 to 3° word)
base['First_Words_N'] = base['Review'].apply(lambda x: len((set(x[0:3]) & set(neg_words))))
# using nltk pack to identify the gramatical class of word
base['Tags'] = base['Review'].apply(lambda x: pos_tag(x))

# counting values of gramatical class
def tag_type(tag):
    tag_list = []
    for i in range(0,len(tag),1):
        tagt = tag[i][1]
        tag_list.append(tagt[0:2])
    return tag_list

base['Tag_Type'] = base['Tags'].apply(tag_type)
base['Dict_Tag'] = base['Tag_Type'].apply(lambda x :{y:x.count(y) for y in x})

# separating gramatical classes in columns
def class_type(tag):
    try:
        adj = tag['JJ']
    except:
        adj = 0
    try:
        vb = tag['VB']
    except:
        vb = 0
    try:
        subs = tag['NN']
    except:
        subs = 0
    try:
        inj = tag['UH']
    except:
        inj = 0
    return pd.Series([adj , vb , subs , inj])

base[['ADJ','VB','SUBS','INJ']] = base['Dict_Tag'].apply(class_type )
# porcentage of positive adjectives
base['ADJ_POS%'] = base['Positive'] / base['ADJ']
# porcentage of negative adjectives
base['ADJ_NEG%'] = base['Negative'] / base['ADJ'] 
# porcentage of adjectives of total words
base['ADJ%'] = base['ADJ'] / base['Total']
# porcentage of verb of total words
base['VB%'] = base['VB'] / base['Total']
# porcentage of substantive of total words
base['SUBS%'] = base['SUBS'] / base['Total']

# adjusting dataframe before scaling
base['ADJ_POS%'] = base['ADJ_POS%'].fillna(0, inplace=True)
base['ADJ_NEG%'] = base['ADJ_NEG%'].fillna(0, inplace=True)
base['Delta'] = base['Delta'].fillna(0, inplace=True)
base['ADJ_POS%'] = base['ADJ_POS%'].replace('inf', 1)
base['ADJ_NEG%'] = base['ADJ_NEG%'].replace('inf', 1)

def ajust(num):
    if num > 1:
        num =1
    else:
        num
    return num

base['ADJ_POS%'] = base['ADJ_POS%'].apply(ajust)
base['ADJ_NEG%'] = base['ADJ_NEG%'].apply(ajust)

# correlation matrix
matrix_corr = base.corr(method='spearman')
matrix_corr = matrix_corr['Rating'].sort_values(ascending=False)

# features
features = base[['Positive%','Negative%','First_Words_P','First_Words_N','ADJ%','Positive']]
# target param
label = base[['Rating']]
# creating scaler
scaler = sklearn.preprocessing.StandardScaler()

features = scaler.fit_transform(features)

# spliting dataset in train and test
train_features, test_features, train_labels, test_labels = train_test_split(features , label, 
                                                                            test_size = 0.25, 
                                                                            random_state = 0)

param_grid = [{'n_estimators':[20,30,40,45,50,55,60,70,100,150,200,250,300,350,400,450,500],
               'max_depth':[5,8,10,12,13,15,16,17,18,19,20,22,25,30,35,40,50,60],
               'criterion':['gini','entropy']}]

clf = sklearn.ensemble.RandomForestClassifier() 

gs = GridSearchCV(clf, param_grid = param_grid, scoring='accuracy', cv=3)

clf.fit(train_features, train_labels)

predictions = clf.predict(test_features)

acc = sklearn.metrics.accuracy_score(test_labels, predictions)