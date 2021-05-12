#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Data is downloaded from https://archive.ics.uci.edu/ml/datasets/Car+Evaluation and converted to car_data.xlsx

# In[2]:


data = pd.read_excel("data/processed/car_data.xlsx")


# ## Exploratory Data Analysis(EDA)

# In[3]:


data.head()


# In[4]:


data.info() # Check for null values


# In[5]:


class_names = set(data['class'])


# In[6]:


# check for unique values of each column
for i in data.columns:
    print(f'{data[i].nunique()}\t{data[i].unique()}')


# Converted string values to integers to make them compatible with scikit learn

# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in data.columns:
    data[i] = le.fit_transform(data[i])
data.head()


# ### Splitting data into training and testing set

# In[8]:


Y = data['class']  # actual output
X = data[data.columns[:-1]]  # input data features
data, target = X, Y
from sklearn.model_selection import train_test_split as SPLIT
X_train, X_test, Y_train, Y_test = SPLIT(X, Y, test_size=0.3, random_state=4)
# 70% Data for Training, 30% Data for Testing


# ### Scale the Data

# In[9]:


from sklearn.preprocessing import StandardScaler as SS

X = SS().fit_transform(X)


# ## Train the Support Vector Classifier

# In[10]:


from sklearn.svm import SVC

# Hyperparameters
kernel = 'rbf'
C = 13
gamma = 0.325

from time import time as T
start = T()
model = SVC(kernel=kernel, C=C, gamma=gamma)
clf = model.fit(X_train, Y_train)
end = T()

pred = clf.predict(X_test)
mScore = clf.score(X_test, Y_test)
print(f'Score against Testing Data: {mScore * 100:.3f}%')
print(f'Model took {(end-start)*1000:.3f}ms to train')


# ### Generate Classification Report

# In[11]:


from sklearn.metrics import classification_report as CR

print("Classification Report:\n",CR(Y_test, pred, zero_division=0))


# ### Cross Validation

# In[12]:


from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import cross_val_score as CVS

model = SVC(kernel='rbf', C=13, gamma=0.325)
folds = 5

start = T()
cross_val = SKF(n_splits=folds, shuffle=True, random_state=4)
scores = CVS(model, X, Y, scoring='accuracy', cv=cross_val)
end = T()

accuracy = scores.mean() * 100
print(f"SVC has mean accuracy of {accuracy:.3f}%\n"
    + f"Cross Validation took {(end-start)*1000:.3f}ms")


# ### Calculate F1-Score of the model

# In[13]:


from sklearn.metrics import f1_score as F1

f1score = F1(Y_test, pred, average = 'weighted')
print(f"SVC has F1-Score = {f1score * 100:.3f}%")


# ### Plot Confusion Matrix

# In[14]:


from sklearn.metrics import plot_confusion_matrix as PCM
PCM(clf, X_test, Y_test);

