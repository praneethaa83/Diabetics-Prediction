#!/usr/bin/env python
# coding: utf-8

# In[2]:


#IMPORING THE REQUIRED LIBRARIES


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
sns.set(rc={'figure.figsize':(10,10)})


# In[4]:


data=pd.read_csv("diabetes_data_upload.csv")


# In[5]:


data


# In[6]:


data.describe()


# In[7]:


data.dtypes


# In[8]:


data.isnull().sum()


# In[9]:


data.info()


# In[10]:


sns.countplot('class',data = data,palette = 'inferno')


# In[11]:


diabetes_positive=data[data['class']=='Positive']
diabetes_negative=data[data['class']=='Negative']
print(diabetes_positive.shape,diabetes_negative.shape)


# In[12]:


male_count=data[data['Gender']=='Male']
female_count=data[data['Gender']=='Female']
print(male_count.shape,female_count.shape)


# In[13]:



sns.countplot(x='Gender',hue='class',data=data)


# In[14]:


sns.countplot(x='Polyuria',hue='class',data=data)


# In[15]:


sns.countplot(x='Polydipsia',hue='class',data=data)


# In[16]:


sns.countplot(x='sudden weight loss',hue='class',data=data)


# In[17]:


sns.countplot(x='weakness',hue='class',data=data)


# In[18]:


sns.countplot(x='Polyphagia',hue='class',data=data)


# In[19]:


sns.countplot(x='Genital thrush',hue='class',data=data)


# In[20]:


sns.countplot(x='visual blurring',hue='class',data=data)


# In[21]:


sns.countplot(x='Itching',hue='class',data=data)


# In[22]:


sns.countplot(x='delayed healing',hue='class',data=data)


# In[23]:


sns.countplot(x='partial paresis',hue='class',data=data)


# In[24]:


sns.countplot(x='Obesity',hue='class',data=data)


# In[25]:


sns.countplot(x='Alopecia',hue='class',data=data)


# In[26]:


pd.crosstab(data.Obesity, data.Age)


# In[27]:


# import preprocessing from sklearn
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
data = data.apply(le.fit_transform)
data.head(520)


# In[ ]:





# In[ ]:





# In[28]:


y=data['class']
X=data.drop('class',axis=1)


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[30]:


#LOGISTIC REGRESSION 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(X_train, y_train)


# In[31]:


y_prediction=model.predict(X_test)
print(y_prediction)


# In[32]:


testscore=model.score(X_test,y_test)
print(testscore)


# In[33]:


#CONFUSION MATRIX
from sklearn.metrics import classification_report, confusion_matrix
conf_m = confusion_matrix(y_test, y_prediction)
print(conf_m)


# In[34]:


report = classification_report(y_test, y_prediction)
print(report)


# In[44]:


#RANDOM FOREST CLASSIFIER
clf = RandomForestClassifier(n_estimators = 100)  
clf.fit(X_train, y_train)


# In[45]:


score2=clf.score(X_test,y_test)
print(score2)


# In[ ]:





# In[38]:


y_pred = clf.predict(X_test)
print(y_pred)


# In[41]:


x=[[41,1,0,1,1,1,0,1,0,1,1,1,0,1,0,0]]
print(clf.predict(x))


# In[49]:


y_pred = clf.predict(X_test)
print(y_pred)


# In[38]:


#KNN CLASSIFIER
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


# In[39]:


ypre=classifier.predict(X_test)
print(ypre)


# In[40]:


score3=classifier.score(X_test,y_test)
print(score3)


# In[41]:


# overfitting of random forest classifier


# In[52]:


#RANDOM FOREST CLASSIFIER
clf = RandomForestClassifier(n_estimators = 100,max_depth=10,min_samples_leaf=5)  
clf.fit(X_train, y_train)


# In[53]:


score2=clf.score(X_test,y_test)
print(score2)


# In[ ]:





# In[54]:


import pickle
Pkl_Filename = "diabetic.pkl"  
pickle.dump(clf, open(Pkl_Filename, 'wb'))


# In[55]:


with open(Pkl_Filename, 'rb') as file:  
    Pickled_RFC_Model = pickle.load(file)

Pickled_RFC_Model


# In[56]:


score = Pickled_RFC_Model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_RFC_Model.predict(X_test)  

Ypredict


# In[ ]:





# In[ ]:




