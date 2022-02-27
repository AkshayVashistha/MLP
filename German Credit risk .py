#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as dtree
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt



# In[2]:


import seaborn as sns


# In[3]:


ak = pd.read_csv("C:\\Users\\loket\\Downloads\\archive\\german_credit_data.csv")


# In[4]:


ak.head()


# In[5]:


ak.dtypes


# In[6]:


ak.isnull().sum()


# In[7]:


ak.shape


# In[8]:


ak.Housing.unique()


# In[9]:


ak.Purpose.unique()


# In[10]:


sns.heatmap(ak.corr(), annot=True).set_title('Correlation Factors Heat Map', color='green', size='30')


# In[11]:


ak['Job'].value_counts()


# In[12]:


sns.set_context('talk', font_scale=.5)
sns.countplot(data=ak , x='Sex', hue='Risk')
plt.show()
sns.catplot(data=ak , x='Sex', y='Age', kind='box')
plt.show()
sns.violinplot(data=ak , x='Sex', y='Age', hue='Risk', split=True)
plt.show()
sns.displot(data=ak , row='Sex', y='Age', col='Risk')
plt.show()


# In[13]:


sns.displot(ak['Age'])
plt.show()


# In[14]:


interval = (18, 25, 35, 60, 120)
cats = ['Young Adult', 'Adult', 'Senior', 'Elder']
ak["Age_cat"] = pd.cut(ak['Age'], interval, labels=cats)


# In[15]:


sns.displot(ak['Credit amount'])
plt.show()


# In[16]:


sns.displot(np.log10(ak['Credit amount']))
plt.show()


# In[18]:


ak['Credit amount'] = np.log10(ak['Credit amount'])


# In[19]:


ak['Saving accounts'] = ak['Saving accounts'].fillna('no_inf')
ak['Checking account'] = ak['Checking account'].fillna('no_inf')


# In[20]:


def one_hot_encoder(ak, column_name, exclude_col = False):
    merged_ak = ak.merge(pd.get_dummies(ak[column_name], drop_first=False, prefix=column_name), left_index=True, right_index=True)
    if exclude_col:
        del merged_ak[column_name] 
    return merged_ak


# In[21]:


ak.columns


# In[22]:


ak = ak.drop('Unnamed: 0', axis='columns')
ak.head()


# In[23]:


ak_ready = ak.copy()


category_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']


# In[24]:


category_features = ['Job','Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk', 'Age_cat']


# In[26]:


for cat in category_features:
    ak_ready = one_hot_encoder(ak_ready, cat, exclude_col=True)
ak_ready.columns


# In[27]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
dataset_ready_x = ak_ready.drop(['Risk_bad', 'Risk_good', 'Age', 'Sex_male'], axis='columns')
X = dataset_ready_x.values
feature_names = dataset_ready_x.columns

y = ak_ready['Risk_bad'].values


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)


# In[30]:


from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, f1_score, precision_score, recall_score
ct = dtree(
           criterion="gini",    # Alternative 'entropy'
           max_depth=None       # Alternative, specify an integer
                              # 'None' means full tree till single leaf
           )
scoring_type = 'accuracy'
kfold = KFold(n_splits=5, random_state=42, shuffle=True) # Ensuring all methods are evaluated on the same fold

score = cross_val_score(ct, X_train, y_train, cv=kfold, scoring=scoring_type)
print(f'Average {scoring_type} performance of the {ct} model = {np.mean(score)}')


# In[31]:


ct.fit(X_train, y_train)
y_pred = ct.predict(X_test)
print(f"Accuracy of our model's prediction {np.sum((y_test == y_pred))/y_test.size}")


# In[32]:


fi = ct.feature_importances_
fi


# In[33]:


list(zip(ak.columns, fi))


# In[ ]:




