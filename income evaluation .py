#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.preprocessing import StandardScaler


# In[4]:


from sklearn.tree import DecisionTreeClassifier as dtree


# In[5]:


from sklearn.tree import export_graphviz


# In[6]:


from sklearn.datasets import load_iris


# In[8]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# In[9]:


import seaborn as sns


# In[10]:


df = pd.read_csv("C:\\Users\\loket\\Downloads\\archive (1)\\income_evaluation.csv")


# In[12]:


df.head()


# In[13]:


df.columns = list(map(lambda a: a.lstrip(), df.columns))


# In[14]:


df.isnull().sum()


# In[15]:


df.shape


# In[16]:


df['workclass'].value_counts()


# In[17]:


shape0 = df.shape[0]
for column in df.columns:
    df[column].replace(' ?', np.NaN, inplace=True)
df = df.dropna().reset_index().drop(columns=['index'])
shape1 = df.shape[0]
print(str(shape0 - shape1) + ' rows have been removed.')


# In[18]:


income = df.income.value_counts()
income


# In[19]:


colors = ['#ADEFD1FF', '#00203FFF']
explode = [0, 0.1]
plt.pie(income, labels=income.values, colors=colors, explode = explode, shadow=True)
plt.title('Income distribution')
plt.legend(labels=income.index)


# In[20]:


df['income'].replace([' <=50K',' >50K'],[1,0], inplace=True)


# In[21]:


df.dtypes


# In[22]:


stats = df.select_dtypes(['float', 'int64']).drop(columns=['income'])


# In[23]:


sns.heatmap(df.corr(), annot=True).set_title('Correlation Factors Heat Map', color='black', size='20')


# In[24]:


df_final = pd.get_dummies(df)
df_final.head()


# In[25]:


X = df_final.drop(columns=['income'])
y = df_final['income']


# In[26]:


ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[27]:


ct = dtree(
           criterion="entropy",    # Alternative 'entropy'
           max_depth=None       # Alternative, specify an integer
                              # 'None' means full tree till single leaf
           )


# In[28]:


_=ct.fit(X_train,y_train)


# In[29]:


y_te = ct.predict(X_test)
np.sum((y_test == y_te))/y_test.size


# In[30]:


fi = ct.feature_importances_
fi


# In[31]:


list(zip(df.columns, fi))


# In[32]:


from sklearn.ensemble import RandomForestClassifier 


# In[33]:


clf=RandomForestClassifier(n_estimators=100)


# In[34]:


clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[35]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




