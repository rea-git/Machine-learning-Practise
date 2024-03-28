#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


# In[3]:


pd.read_csv("magic04.data")


# In[5]:


col=['Flength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fM3Alpha','fDist','class']
df=pd.read_csv("magic04.data",names=col)
df.head()


# In[6]:


df["class"].unique()


# In[7]:


df["class"]=(df["class"] == "g").astype(int)


# In[8]:


df["class"].unique()


# In[9]:


df.head()


# In[10]:


print(len(df))


# In[60]:


for label in col[:-1]:
    plt.hist(df[df["class"]==1][label],color='blue',label='gamma',alpha=0.5,density=True)
    plt.hist(df[df["class"]==0][label],color='Red',label='hedron',alpha=0.5,density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()


# In[11]:


#Train,valid,test dataset
train, valid, test = np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])


# In[12]:


def scale_dataset(dataframe, oversample=False):
    x=dataframe[dataframe.columns[:-1]].values
    y=dataframe[dataframe.columns[-1]].values
    
    scaler=StandardScaler()
    x=scaler.fit_transform(x)
    
    if oversample:
        ros=RandomOverSampler()
        x,y=ros.fit_resample(x,y)
    data=np.hstack((x,np.reshape(y,(-1,1))))
    return data,x,y


# In[13]:


print(len(train[train["class"]==1]))#gamma
print(len(train[train["class"]==0]))#hedron


# In[14]:


train,x_train,y_train = scale_dataset(train,oversample=True)


# In[15]:


print(len(y_train))


# In[16]:


print(len(x_train))


# In[17]:


print(len(train))


# In[ ]:




