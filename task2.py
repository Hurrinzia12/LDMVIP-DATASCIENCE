#!/usr/bin/env python
# coding: utf-8

# ## **Author-HURRIN ZIA**
# ## **Data Science intern at Letsgrowmore**
# ## **Batch: December 2021**
# ## **Task3: Prediction using Decision Tree Algorithm**
# ## **Aim: Create the Decision Tree classifier and visualize it graphically**
# ## **Level: Intermediate**
# ## **Dataset Link:https://drive.google.com/file/d/11Iq7YvbWZbt8VXjfm06brx66b10YiwK-/view**
# ## **Importing all necessary libraries to perform the task**

# In[1]:


import numpy as np
import pandas as pd
import sklearn.metrics as sm
import seaborn as sns
import matplotlib.pyplot as mt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# ##  Importing the iris data set

# In[4]:


data=pd.read_csv("C:\\Users\\HURRINZIA\\Downloads\\Iris.csv")
data.head()


# ## Getting the size of the Iris Data

# In[5]:


data_size= data.shape
print(f"Number of Rows in Iris data : {data_size[0]}") 
print(f"Number of Columns in Iris data : {data_size[1]}")


# ## Checking the Information of the Iris Data.

# In[6]:


data.info()


# ## Checking the null values.

# In[7]:


data.isnull()


# In[8]:


data.isnull().sum()


# ## Describing the Statistical measures of the Dataset.

# In[9]:


df=data.drop('Id', axis=1)
df.describe()


# ## Data Preparation

# In[10]:


target=data['Species']
df=data.copy()
df=data.drop('Species', axis=1)
data.shape


# In[11]:


X=data.iloc[:, [0,1,2,3]].values
le=LabelEncoder()
data['Species']=le.fit_transform(data['Species'])
y=data['Species'].values
data.shape


# ## Splitting dataset into Train and Test sets.

# In[12]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Traingin split:",X_train.shape)
print("Testin spllit:",X_test.shape)


# ## After Splitting data into test and train now going to defining the decision tree algorithm.

# In[13]:


decision_tree=DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
print("Decision Tree Classifier created!")


# ## After making decision tree classifier now create classification report and confusion matrix.

# In[14]:


y_pred=decision_tree.predict(X_test)
print("Classification report:\n",classification_report(y_test,y_pred))


# ###  For this model, the accuracy on the test set is 1, i.e, 100%,
# ### which means the model made the right prediction for 100% of the Iris in the given dataset.
# ### Now creating Confusion metrix

# In[15]:


import numpy as np
confusion_matrix=confusion_matrix(y_test,y_pred)
confusion_matrix


# ## Now visualizing the trained model that we have created.

# In[16]:


mt.figure(figsize=(20,10))
tree=plot_tree(decision_tree,feature_names=df.columns,precision=2,rounded=True,filled=True,class_names=target.values)

