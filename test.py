import statistics
import numpy as np
import pandas as pd
import seaborn as sns
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
from function import *
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


def t(model1_f,model2_f,model3_f):
    data = pd.read_csv('airline-test-samples.csv')
    ##########################################################
    X = data.iloc[:, 0:10]
    Y = data.iloc[:, -1]
    X=preprocessing_x(X)
    Y=preprocessing_y(Y)
    ##########################################################
    model1 = pickle.load(open('model1', 'rb'))
    top_feature1=model1_f####
    X = X[top_feature1]
    poly_features1 = PolynomialFeatures(degree=2)
    X_train_poly1 = poly_features1.fit_transform(X)
    result1 = model1.score(X_train_poly1, Y)
    print("model1",result1)

    model2 = pickle.load(open('model2', 'rb'))
    top_feature2=model2_f####
    X = X[top_feature2]
    poly_features2 = PolynomialFeatures(degree=3)
    X_train_poly2 = poly_features2.fit_transform(X)
    result2 = model2.score(X_train_poly2, Y)
    print("model2",result2)
    
    model3 = pickle.load(open('model3', 'rb'))
    top_feature3=model3_f####
    X = X[top_feature3]
    poly_features3 = PolynomialFeatures(degree=4)
    X_train_poly3 = poly_features3.fit_transform(X)
    result3 = model3.score(X_train_poly3, Y)
    print("model3",result3)
    
    model4 = pickle.load(open('model4', 'rb'))
    result4 = model4.score(X, Y)
    print("model4",result4)

