# Cancer-cell-Prediction
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn import metrics


# In[2]:


data=pd.read_csv('cell_samples.csv')


# In[3]:


data.head(20)
data.count()
data['Class'].value_counts()


# In[4]:


malignant_df=data[data['Class']==4][:200]
benign_df=data[data['Class']==2][:200]


# In[5]:



axes=benign_df.plot(kind='scatter', x='Clump', y='UnifSize', color='blue', label='BENIGN')
malignant_df.plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='MALIGNANT', ax=axes)


# In[6]:


data.dtypes


# In[7]:


data=data[pd.to_numeric(data['BareNuc'],errors='coerce').notnull()]

data['BareNuc']=data['BareNuc'].astype(int)


# In[8]:


data.dtypes


# In[9]:


data.head()


# In[10]:


data.columns


# In[11]:


X=data[['Clump', 'UnifSize', 'UnifShape','MargAdh', 'SingEpiSize', 'BareNuc' ,'BlandChrom','NormNucl','Mit']]
y=data['Class']


# In[26]:



x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(X,y,test_size=0.2)

clf=svm.SVC(kernel='linear',C=8)

clf.fit(x_train,y_train)


# In[27]:



y_pred=clf.predict(x_test)
acc=metrics.accuracy_score(y_test,y_pred)
acc


# In[28]:


y_test=list(y_test)
x_test=pd.DataFrame(x_test)
x_test.values


# In[29]:


data.head()


# In[30]:


#Cheking model accuracy by comparing actual known results with the predicted ones 
for i in range(len(y_pred)):
    print(y_pred[i],x_test.values[i],y_test[i])


# In[41]:


x_input=[]


# In[42]:


def inputt():
    print('Enter details about cells in the following order :clump ->UnifSize -> UnifShape->MargAdh ->SingEpiSize -> BareNuc->BlandChrom-> NormNucl->Mit')
    x_input=input().split()


# In[43]:


def predict():
    inputt()
    x_predict=clf.predict(x_input)
    if(x_predict==2):
        print('Benign cell')
    elif(x_predict==4):
        print('Malignant cell')


# In[44]:


Ch='y'
while(Ch=='y'):
    print('*********************************************CANCER CELL PREDICTION*****************************')
    print()
    predict()
    print('Do you want to comntinue?(y/n)')
    Ch=input()


# In[ ]:





# In[ ]:




