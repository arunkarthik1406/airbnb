import  pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


os.chdir('/users/arunkarthik/Downloads')

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)

data = pd.read_csv('airbnb.csv')
print(data.head())
print(data.info())
print(data.describe())

# data Pre-processing

print('\n\t\tOnly objects \n',data.select_dtypes(include=['object']).head())
print('\n\t\t Only numeric \n',data.select_dtypes(exclude=['object']).head())

# changing non-numeric objects to numeric

data['last_review'] = pd.to_datetime(data['last_review'])
print(data['last_review'].dtype)

print(data['neighbourhood_group'].unique())

#====================#============================#=========================#
# segmented analysis

'''Brooklyn Data'''

Brooklyn_data= data[data['neighbourhood_group']=='Brooklyn']
print(Brooklyn_data['neighbourhood'].nunique())


#====================#============================#=========================#

'''Manhattan Data'''
Manhattan_data = data[data['neighbourhood_group']=='Manhattan']
plt.figure(figsize=(14,7))
Manhattan_data.plot.scatter(x='latitude',y='longitude',c='green')

plt.figure(figsize=(14,7))
sns.distplot(Manhattan_data['price'],bins=30)

plt.figure(figsize=(14,7))
sns.barplot(x=Manhattan_data['neighbourhood'],y=Manhattan_data['price'])

plt.figure(figsize=(14,7))
sns.barplot(x=Manhattan_data['neighbourhood'],y=Manhattan_data['price'])
plt.show()
#====================#============================#=========================#

'''Queens Data'''
Queens_data = data[data['neighbourhood_group']=='Queens']

#====================#============================#=========================#

'''Manhattan Data'''
Staten_Island_data = data[data['neighbourhood_group']=='Staten Island']

#====================#============================#=========================#

'''Bronx Data'''
Bronx_data = data[data['neighbourhood_group']=='Bronx']