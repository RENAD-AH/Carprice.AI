#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# In[3]:


df=pd.read_csv("CarPrice.csv")


# In[5]:


df.info()


# In[6]:


df.nunique()


# In[7]:


df.CarName.unique()


# In[8]:


df["CarName"].value_counts()


# In[9]:


df['CarName']=df['CarName'].str.split(' ',expand=True)[0]


# In[10]:


df['CarName'].unique()


# In[11]:


df['CarName']=df['CarName'].replace({'maxda':'mazda','nissan':'Nissan','porcshce':'porsche','toyouta':'toyota','vokswagen':'volkswagen','vw':'vilkswagen'})


# In[12]:


df['CarName'].unique()


# In[13]:


plt.figure(figsize=(15,15))
ax=sns.countplot(x=df["CarName"]);
ax.bar_label(ax.containers[0]);
plt.xticks(rotation=90);


# In[14]:


sns.set_style("whitegrid")
plt.figure(figsize=(15,10))
sns.distplot(df.price)
plt.show()


# In[15]:


ax=sns.countplot(x=df["fueltype"]);
ax.bar_label(ax.containers[0]);


# In[19]:


import seaborn as sns
sns.pairplot(df,markers=None,hue='price')
plt.show()


# In[22]:


new_df = df[['fueltype','aspiration','doornumber','carbody','drivewheel','enginetype','cylindernumber','fuelsystem','wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg','price']]
new_df.head()


# In[23]:


new_df=pd.get_dummies(columns=["fueltype","aspiration","doornumber","carbody","drivewheel","enginetype","cylindernumber","fuelsystem"],data=new_df)


# In[25]:


scaler=StandardScaler()
num_cols=['wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg']
new_df[num_cols]=scaler.fit_transform(new_df[num_cols])


# In[26]:


x=new_df.drop(columns=["price"])
y=new_df["price"]
x.shape


# In[27]:


y.shape


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[30]:


training_score=[]
testing_score=[]


# In[31]:


from sklearn.metrics import r2_score

def model_prediction(model):
    model.fit(x_train,y_train)
    x_train_pred=model.predict(x_train)
    x_test_pred=model.predict(x_test)
    a=r2_score(y_train,x_train_pred)*100
    b=r2_score(y_test,x_test_pred)*100
    training_score.append(a)
    testing_score.append(b)
    
    print(f"r2_score of {model} model on Training Data is:",a )
    print(f"r2_score of {model} model on Testing Data is:",b)


# In[34]:


from sklearn.linear_model import LinearRegression
model_prediction(LinearRegression())


# In[37]:


from sklearn.tree import DecisionTreeRegressor
model_prediction(DecisionTreeRegressor())


# In[38]:


from sklearn.ensemble import RandomForestRegressor
model_prediction(RandomForestRegressor())


# In[42]:


#!pip install catboost


# In[43]:


from catboost import CatBoostRegressor
model_prediction(CatBoostRegressor(verbose=False))


# In[44]:


models=["Linear Regressor","Decision Tree","Random Forest","CatBoost"]


# In[45]:


df1=pd.DataFrame({"Algorithms":models,
                 "Training Score":training_score,
                 "Testing Score":testing_score})
df1


# In[47]:


df1.plot(x='Algorrithms',y=['Training Score','Testing Score'],figsize=(16,6),kind='bar', title='Performance Visualهzation of Different Models',colormap='Set1')


# In[ ]:




