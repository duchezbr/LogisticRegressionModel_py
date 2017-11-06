# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:29:53 2017

@author: duche
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Breast Cancer Wisconsin (diagnostic) data set from the UCI Machine Learning Repository

bc = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')

# The response variable is found in the second column
response = bc.iloc[:, 1]

# 'M' will be equal to 1 and 'B' will be equal to 0.
label = []
for i in range(len(response)):
    if response.loc[i] == 'M':
        label.append(1)
    else:
        label.append(0)
        
label = np.array(label, dtype = float)
# remove the sample number (column 0) and response variable (column 1) to return a data frame containing only predictors
 
bc = bc.drop(bc.columns[[0, 1]], axis = 1)

# Standardize predictor varialbes (mean = 0, sd = 1) 
mu = bc.mean()
sd = bc.std()
numerator = bc - mu
normBC = numerator/sd

#%%
# change normBC to np array befor running logistic regression model
normBC = normBC.values
# obtain dimensions of the array
rows = len(normBC)
coefficients = len(normBC[0, :])+1
# create zeros array to fill with coefficients
beta = np.zeros((1, coefficients))

#%%
# We will use the Logistic Regression Model provided here to train our model.  We will set our alpha value to 0.3 and iterate throuh our data 10 times initially.  These values can be changed on successive training attempts to improve the accuracy of our model.


alpha = 0.3
epoch = 10
accuracy = np.empty([epoch, 1], dtype = float)


for j in range(epoch):
    
    # score variable will be populated with predicted values for the response           variable
    score = np.zeros((rows, 1))
    correctlyIdentified = 0
    
    for i in range(rows):
        
        # obtain a predicted value between 0 and 1
        prediction = 1/(1 + np.exp(-(beta[0, 0] + sum(beta[0, 1::]*normBC[i, :]))))
        
        if prediction < 0.5:
                
            score[i] = 0
        else:
            score[i] = 1
         
        # adjust beta coefficients based on accuracy of the prediction
        beta[0, 0] = beta[0, 0] + alpha*(label[i] - prediction)*prediction*(1-prediction)*1
        
        for k in range(1, coefficients):
         
            beta[0, k] = beta[0, k] + alpha*(label[i] - prediction)*prediction*(1-prediction)*normBC[i, k-1]
    
    # after each epoch determine the number of correct predictions
    for m in range(rows):
        if score[m] == label[m]:
            correctlyIdentified += 1.0        
    
    accuracy[j] = correctlyIdentified/rows
    
# plot the percentage of correct predictions after each epoch
plt.plot(accuracy)
        

