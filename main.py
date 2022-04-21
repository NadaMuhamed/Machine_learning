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
import re
import time
##########################################################
data = pd.read_csv('airline-price-prediction.csv')
data.dropna(how='any',inplace=True)

##########################################################
X=data.iloc[:,0:10]
# Y= [float(f) for f in data['price']]
Y = handel_price(data['price'])
#[int(y) for y in handel_price(data['price'])]
# airline= data.iloc[:,:]
# airline['price'] = Y
###################  'route'
X = dictionary_to_columns(X, 'route')
cols=('airline','ch_code','type', 'source', 'destination')
X = Feature_Encoder(X,cols)
################### timestamp for 'date'
X = Date_Converter(X)
################### 'time_taken'
X['time_taken'] = time_taken_to_seconds(X)
###################### 'stop'
X['stop'] = Stop_Feature(X['stop'])
####################converttomin
X['dep_time']= converttomin(X['dep_time'])
X['arr_time']= converttomin(X['arr_time'])
############################
airline = X
airline['price'] = Y


###########################"Model 1"###############################
start1=time.time()
print("\n  Model 1  \n")
corr = airline.corr()
top_feature1 = corr.index[abs(corr['price'])>0.1]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr1 = airline[top_feature1].corr()
sns.heatmap(top_corr1, annot=True)
#plt.show()
top_feature1 = top_feature1.delete(-1)
X = X[top_feature1]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y, test_size = 0.50,shuffle=True,random_state=10)

poly_features1 = PolynomialFeatures(degree=2)
X_train_poly1 = poly_features1.fit_transform(X_train1)
poly_model1 = linear_model.LinearRegression()
poly_model1.fit(X_train_poly1, y_train1)
y_train_predicted1 = poly_model1.predict(X_train_poly1)
ypred1=poly_model1.predict(poly_features1.transform(X_test1))
prediction1 = poly_model1.predict(poly_features1.fit_transform(X_test1))
#print('Co-efficient of linear regression',poly_model1.coef_)
#print('Intercept of linear regression model',poly_model1.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test1, prediction1))
print('regression score function model 1', r2_score(y_test1, prediction1))
true_price1=np.asarray(y_test1)[0]
predicted_pruce1=prediction1[0]
print('True price for the test set is : ' + str(true_price1))
print('Predicted price for the test set is : ' + str(predicted_pruce1))
exec1=time.time()-start1
print("Training Time for Model 1   :   ",exec1)

###########################"Model 2"###############################
start2=time.time()
print("\n  Model 2  \n")
top_feature2 = corr.index[abs(corr['price'])>0.3]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr2 = airline[top_feature2].corr()
sns.heatmap(top_corr2, annot=True)
#plt.show()
top_feature2 = top_feature2.delete(-1)
X = X[top_feature2]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y, test_size = 0.30,shuffle=True,random_state=10)

poly_features2 = PolynomialFeatures(degree=3)
X_train_poly2 = poly_features2.fit_transform(X_train2)
poly_model2 = linear_model.LinearRegression()
poly_model2.fit(X_train_poly2, y_train2)
y_train_predicted2 = poly_model2.predict(X_train_poly2)
ypred2=poly_model2.predict(poly_features2.transform(X_test2))
prediction2 = poly_model2.predict(poly_features2.fit_transform(X_test2))
#print('Co-efficient of linear regression',poly_model2.coef_)
#print('Intercept of linear regression model',poly_model2.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test2, prediction2))
print('Mean Square Error', metrics.mean_squared_error(y_test2, prediction2))
print('regression score function model 2', r2_score(y_test2, prediction2))
true_price2=np.asarray(y_test2)[0]
predicted_pruce2=prediction2[0]
print('True price for the test set is : ' + str(true_price2))
print('Predicted price for the test set is : ' + str(predicted_pruce2))
exec2=time.time()-start2
print("Training Time for Model 2   :   ",exec2)
###########################"Model 3"###############################
start3=time.time()
print("\n  Model 3  \n")
top_feature3 = corr.index[abs(corr['price'])>0.3]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr3 = airline[top_feature3].corr()
sns.heatmap(top_corr3, annot=True)
#plt.show()
top_feature3 = top_feature3.delete(-1)
X = X[top_feature3]

X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Y, test_size = 0.40,shuffle=True,random_state=10)

poly_features3 = PolynomialFeatures(degree=4)
X_train_poly3 = poly_features3.fit_transform(X_train3)
poly_model3 = linear_model.LinearRegression()
poly_model3.fit(X_train_poly3, y_train3)
y_train_predicted3 = poly_model3.predict(X_train_poly3)
ypred3=poly_model3.predict(poly_features3.transform(X_test3))
prediction3 = poly_model3.predict(poly_features3.fit_transform(X_test3))
#print('Co-efficient of linear regression',poly_model3.coef_)
#print('Intercept of linear regression model',poly_model3.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test3, prediction3))
print('regression score function model 3', r2_score(y_test3, prediction3))
true_price3=np.asarray(y_test3)[0]
predicted_pruce3=prediction3[0]
print('True price for the test set is : ' + str(true_price3))
print('Predicted price for the test set is : ' + str(predicted_pruce3))
exec3=time.time()-start3
print("Training Time for Model 3   :   ",exec3)

###########################"Model 4"###############################
print("\n  Model 4  \n")
start4=time.time()
linear_model4 = linear_model.LinearRegression()
linear_model4.fit(X,Y)
prediction4= linear_model4.predict(X)
#print('Co-efficient of linear regression',linear_model4.coef_)
#print('Intercept of linear regression model',linear_model4.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y), prediction4))
print('regression score function model 4', r2_score(Y, prediction4))
exec4=time.time()-start4
print("Training Time for Model 4   :   ",exec4)
####################################################################################################################
