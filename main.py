import numpy as np
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
from function import *
import statistics
from nltk import DecisionTreeClassifier
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import *
import re
import time
from statistics import mean
from fractions import Fraction
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pickle
from function import preprocessing_x
from test import *
##########################################################
data = pd.read_csv('airline-price-prediction.csv')
#data.dropna(how='any', inplace=True)

##########################################################
X = data.iloc[:, 0:10]
Y = data.iloc[:, -1]
X =preprocessing_x(X)
Y =preprocessing_y(Y)
airline = X
airline['price'] = Y

###########################"Model 1"###############################
corr = airline.corr()
top_feature1 = corr.index[abs(corr['price']) > 0.1]
top_corr1 = airline[top_feature1].corr()
top_feature1 = top_feature1.delete(-1)
###########################"Model 2"###############################
top_feature2 = corr.index[abs(corr['price']) > 0.3]
top_corr2 = airline[top_feature2].corr()
top_feature2 = top_feature2.delete(-1)
###########################"Model 3"###############################
top_feature3 = corr.index[abs(corr['price']) > 0.3]
top_corr3 = airline[top_feature3].corr()
top_feature3 = top_feature3.delete(-1)
###############################################################################################
t(top_feature1, top_feature2, top_feature3)
