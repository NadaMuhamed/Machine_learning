import ast
import math
from datetime import timedelta, datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
from sklearn import svm, datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import *
import time
import ast
import math
from datetime import timedelta, datetime
from sklearn.preprocessing import LabelEncoder
import re


def time_taken_to_seconds(X):
    UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
    return [int(timedelta(**{
        UNITS.get(m.group('unit').lower(), 'seconds'): float(m.group('val'))
        for m in re.finditer(r'(?P<val>\d+(\.\d+)?)(?P<unit>[smhdw]?)', s, flags=re.I)
    }).total_seconds()) for s in X['time_taken']]



def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def dictionary_to_columns(X, colmn):
    temp = X[colmn].values
    X.drop(colmn, inplace=True, axis=1)
    s = [ast.literal_eval(d)['source'] for d in temp]
    d = [ast.literal_eval(d)['destination'] for d in temp]
    X['source'] = s
    X['destination'] = d
    return X

### date
def Date_Converter(X):
    datalist = [datetime.timestamp(datetime.strptime(d, '%d/%m/%Y')) for d in [t.replace('-', '/') for t in X['date'].values]]
    X['date'] = datalist
    temp=[int(str(p)[:-4]) for p in X['date'].values]
    X['date']=temp

    return X


def Stop_Feature(column):
    values = ["1stop", "nonstop", "2stop"]
    spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                  "*", "+", ",", "-", ".", "/", ":", ";", "<",
                  "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                  "`", "{", "|", "}", "~", "–"]
    for char in spec_chars:
        column = column.str.replace(char, '', regex=True)

    column = column.replace(values[0], 1, regex=True)
    column = column.replace(values[1], 0, regex=True)
    column = column.replace(values[2], 2, regex=True)
    return column



def converttomin(x):
    x=x.str.split(':').apply(lambda x: int(x[0]) * 60*60 + int(x[1])*60)
    return x


def handel_price(Y):
    return [ int(f) for f in[ t.replace(',','') for t in Y]]


def preprocessing_x(X):
    X = dictionary_to_columns(X, 'route')
    cols = ('airline', 'ch_code', 'type', 'source', 'destination')
    X = Feature_Encoder(X, cols)
    X = Date_Converter(X)
    X['time_taken'] = time_taken_to_seconds(X)
    X['stop'] = Stop_Feature(X['stop'])
    X['dep_time'] = converttomin(X['dep_time'])
    X['arr_time'] = converttomin(X['arr_time'])
    X['date'].fillna(16466895.662425898, inplace=True)
    X['airline'].fillna(3.739671451408779, inplace=True)
    X['ch_code'].fillna(4.28735512555785, inplace=True)
    X['num_code'].fillna(1422.2229026510358, inplace=True)
    X['dep_time'].fillna(48338.42794578032, inplace=True)
    X['time_taken'].fillna(43938.79031506028, inplace=True)
    X['stop'].fillna(0.9236828082328649, inplace=True)
    X['arr_time'].fillna(56646.54008192899, inplace=True)
    X['type'].fillna(0.6892443215879571, inplace=True)
    X['source'].fillna(2.5745395657097183, inplace=True)
    X['destination'].fillna(2.5873326450409646, inplace=True)
    return X

def preprocessing_y(Y):
    Y.fillna(1105, inplace=True)
    Y = handel_price(Y)
    return Y





