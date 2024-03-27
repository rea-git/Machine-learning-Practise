#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


pd.read_csv("magic04.data")


# In[34]:


col=['Flength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fM3Alpha','fDist','class']
df=pd.read_csv("magic04.data",names=col)
df.head()


# In[35]:


df["class"].unique()


# In[36]:


df["class"]=(df["class"] == "g").astype(int)


# In[37]:


df["class"].unique()


# In[38]:


df.head()


# In[42]:


for label in col[:-1]:
    plt.hist(df[df["class"]==1][label],color='blue',label='gamma',alpha=0.5,density=True)
    plt.hist(df[df["class"]==0][label],color='Red',label='hedron',alpha=0.5,density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()


# In[43]:


#Train,valid,test dataset
train, valid, test = np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])


# In[ ]:




