# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:23:34 2019

@author: Aditya Kumar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#loading the dataset that is required
dataset = pd.read_csv('cpcb_dly_aq_rajasthan-2005.csv')
dataset = dataset.loc[dataset['City/Town/Village/Area'] == 'Jaipur']
features  = dataset.iloc[:, [0, 3, 4, 5, 6, 7]]

dataset_2 = pd.read_csv('cpcb_dly_aq_rajasthan-2006.csv')
dataset_2 = dataset_2.loc[dataset_2['City/Town/Village/Area'] == 'Jaipur']
features_2  = dataset_2.iloc[:, [0, 3, 4, 5, 6, 7]]


dataset_3 = pd.read_csv('cpcb_dly_aq_rajasthan-2007.csv')
dataset_3 = dataset_3.loc[dataset_3['City/Town/Village/Area'] == 'Jaipur']
features_3  = dataset_3.iloc[:, [0, 3, 4, 5, 6, 7]]

dataset_4 = pd.read_csv('cpcb_dly_aq_rajasthan-2008.csv')
dataset_4 = dataset_4.loc[dataset_4['City/Town/Village/Area'] == 'Jaipur']
features_4  = dataset_4.iloc[:, [0, 3, 4, 5, 6, 7]]


dataset_5 = pd.read_csv('cpcb_dly_aq_rajasthan-2009.csv')
dataset_5 = dataset_5.loc[dataset_5['City/Town/Village/Area'] == 'Jaipur']
features_5  = dataset_2.iloc[:, [0, 3, 4, 5, 6, 7]]


dataset_6 = pd.read_csv('cpcb_dly_aq_rajasthan-2010.csv')
dataset_6 = dataset_6.loc[dataset_6['City/Town/Village/Area'] == 'Jaipur']
features_6  = dataset_6.iloc[:, [1, 4, 6, 7, 8, 9]]


dataset_7 = pd.read_csv('cpcb_dly_aq_rajasthan-2011.csv')
dataset_7 = dataset_7.loc[dataset_7['City/Town/Village/Area'] == 'Jaipur']
features_7  = dataset_7.iloc[:, [1, 4, 6, 7, 8, 9]]


dataset_8 = pd.read_csv('cpcb_dly_aq_rajasthan-2012.csv')
dataset_8 = dataset_8.loc[dataset_8['City/Town/Village/Area'] == 'Jaipur']
features_8  = dataset_8.iloc[:, [1, 4, 6, 7, 8, 9]]


dataset_9 = pd.read_csv('cpcb_dly_aq_rajasthan-2013.csv')
dataset_9 = dataset_9.loc[dataset_9['City/Town/Village/Area'] == 'Jaipur']
features_9  = dataset_9.iloc[:, [1, 4, 6, 7, 8, 9]]


dataset_10 = pd.read_csv('cpcb_dly_aq_rajasthan-2014.csv')
dataset_10 = dataset_10.loc[dataset_10['City/Town/Village/Area'] == 'Jaipur']
features_10  = dataset_10.iloc[:, [1, 4, 6, 7, 8, 9]]


dataset_11 = pd.read_csv('cpcb_dly_aq_rajasthan-2015.csv')
dataset_11 = dataset_11.loc[dataset_11['City/Town/Village/Area'] == 'Jaipur']
features_11  = dataset_11.iloc[:, [1, 4, 6, 7, 8, 9]]

#including all features sets in a new dataset
new = pd.concat([features, features_2, features_3, features_4, features_5, features_6, features_7, features_8, features_9, features_10, features_11])

#filling nan values with mean to make new dataset
final = new.fillna(new.mean()) 
final = final.reset_index(drop= True)

#summary of data for every location
sum_vkia = final.loc[final['Location of Monitoring Station'] == 'VKIA']
vkia  = sum_vkia.iloc[:,[3,4,5]]
vkia_sum = vkia.describe()
print(vkia_sum)

sum_mia = final.loc[final['Location of Monitoring Station'] == 'MIA']
mia  = sum_mia.iloc[:,[3,4,5]]
mia_sum = mia.describe()
print(mia_sum)

sum_ajmeri = final.loc[final['Location of Monitoring Station'] == 'Ajmeri Gate']
ajmeri  = sum_ajmeri.iloc[:,[3,4,5]]
ajmeri_sum = ajmeri.describe()
print(ajmeri_sum)

sum_chand = final.loc[final['Location of Monitoring Station'] == 'Chandpole']
chand  = sum_chand.iloc[:,[3,4,5]]
chand_sum = chand.describe()
print(chand_sum)

sum_rspcb = final.loc[final['Location of Monitoring Station'] == 'RSPCB Office']
rspcb  = sum_rspcb.iloc[:,[3,4,5]]
rspcb_sum = rspcb.describe()
print(rspcb_sum)

sum_vid = final.loc[final['Location of Monitoring Station'] == 'Vidyadhar Nagar']
vid  = sum_vid.iloc[:,[3,4,5]]
vid_sum = vid.describe()
print(vid_sum)

#correlation matrix for each location
def col_vkia(vkia):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1  = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(vkia.corr(),interpolation="nearest", cmap=cmap)
    ax1.grid=(True)
    plt.title('Correlation Matrix for VKIA')
    plt.show()
col_vkia(vkia)    

def col_mia(mia):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1  = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(mia.corr(),interpolation="nearest", cmap=cmap)
    ax1.grid=(True)
    plt.title('Correlation Matrix for MIA')
    plt.show()
col_vkia(mia)

def col_ajmeri(ajmeri):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1  = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(ajmeri.corr(),interpolation="nearest", cmap=cmap)
    ax1.grid=(True)
    plt.title('Correlation Matrix for Ajmeri Gate')
    plt.show()
col_ajmeri(ajmeri)

def col_chand(chand):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1  = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(chand.corr(),interpolation="nearest", cmap=cmap)
    ax1.grid=(True)
    plt.title('Correlation Matrix for Chandpole')
    plt.show()
col_chand(chand)

def col_rspcb(rspcb):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1  = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(rspcb.corr(),interpolation="nearest", cmap=cmap)
    ax1.grid=(True)
    plt.title('Correlation Matrix for RSPCB Office')
    plt.show()
col_rspcb(rspcb)

def col_vid(vid):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1  = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(vid.corr(),interpolation="nearest", cmap=cmap)
    ax1.grid=(True)
    plt.title('Correlation Matrix for Vidyadhar Nagar')
    plt.show()
col_vid(vid)

 

data = final.iloc[:, [0,3,4,-1]]
month = []
year = []
date = []
for item in data['Sampling Date']:
    month_1 = []
    month_1 = item.split("-")
    year.append(int(month_1[2]))
    month.append(int(month_1[1]))
    date.append(int(month_1[0]))

date = pd.DataFrame(date)    
year = pd.DataFrame(year) 
month = pd.DataFrame(month)   
data['Month'] = month
data['Year'] = year
data['Date'] = date
data_1 = data.drop('Sampling Date', axis = 1)

#prepare the data to train the model
features = data_1.iloc[:, 3:6]
labels = data_1.iloc[:, 0:3]

#train test split
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)  


#train the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=25, random_state=0)  
regressor.fit(features_train, labels_train)  
labels_pred = regressor.predict(features_test) 
labels_pred = labels_pred.round(2)
labels_test = labels_test.round(2)

labels_pred = pd.DataFrame(labels_pred)

import pickle
with open('Picklefile.pkl', 'wb') as f:
    pickle.dump(regressor,f)    

from sklearn.metrics import accuracy_score
print(accuracy_score(labels_pred,labels_test))

#evaluating the model
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, labels_pred))  

#to take input date
input_data = [5,1996,13]
import numpy as np
input_data = np.array(input_data)
input_data = input_data.reshape(1,3)
input_pred = regressor.predict(input_data)