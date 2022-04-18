import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import *
from function import *
import re

##########################################################
data = pd.read_csv('airline-price-prediction.csv')
data.dropna(how='any',inplace=True)

##########################################################
X=data.iloc[:,0:10]
Y=data['price']
airline= data.iloc[:,:]
###################  'route'
X = dictionary_to_columns(X, 'route')
cols=('airline','ch_code','type', 'source', 'destination')
X = Feature_Encoder(X,cols)
corr = airline.corr()
################### timestamp for 'date'
X = Date_Converter(X)
################### 'time_taken'
X['time_taken'] = time_taken_to_seconds(X)
###################### 'stop'
X['stop'] = Stop_Feature(X['stop'])


####################converttomin
X['dep_time']= converttomin(X['dep_time'])
X['arr_time']= converttomin(X['arr_time'])

