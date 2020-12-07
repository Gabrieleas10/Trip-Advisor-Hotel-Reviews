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

# creating dataset balaced
base = pd.concat([df_maj , dfm1 , dfm2 , dfm3 , dfm4])
