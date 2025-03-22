# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 19:50:37 2025

@author: ADMIN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds=pd.read_csv(r'c:\Users\ADMIN\Downloads\Salary_Data.csv')
print(ds)

print(ds.shape)

x=ds.iloc[:,:-1]
y=ds.iloc[:,-1]
print(x,y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,test_size=0.3,random_state=0)

y_test=y_test.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)



plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()

#best fit line hear
coef=print(f"Coeficient:{regressor.coef_}")
intercept=print(f'Intercept:{regressor.intercept_}')



#whose age is 12yrs (future prediction)
exp_12_future_pred=9360*12+26777
exp_12_future_pred

bias=regressor.score(x_train,y_train)
print(bias)

variance=regressor.score(x_test,y_test)
print(variance)

#we can impletent statistic to this dataset'
print(ds.mean())
print(ds['Salary'].mean())

print(ds.median())
print(ds['Salary'].median())

print(ds.mode())
print(ds['Salary'].mode())

print(ds.var())
print(ds['Salary'].var())

print(ds.std())
print(ds['Salary'].std())

from scipy.stats import variation
print(variation(ds.values))
print(variation(ds['Salary']))

#correlation
print(ds.corr())
print(ds['Salary'].corr(ds['YearsExperience']))

print(ds.skew())
print(ds['Salary'].skew()) #this will give the skewness of the particular attribute

print(ds.sem()) #his will give the standard error of the data set
print(ds['Salary'].sem())

#zscore
import scipy.stats as stats
print(ds.apply(stats.zscore)) #it gives the zscore of the entire dataset

print(stats.zscore(ds['Salary']))

a=ds.shape[0]
b=ds.shape[1]

degree_of_freedom=a-b
print(degree_of_freedom)

y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

mean_total=np.mean(ds.values)
sst=np.sum((ds.values-mean_total)**2)
print(sst)

#r2 square

r_square=1-(SSR/sst) #range of r square is 0 to 1 
print(r_square)
