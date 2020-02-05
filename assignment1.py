#!/usr/bin/env python
# coding: utf-8

# 1.
# a) To predict the  pokemon is legendary or not based on given varaiables
# b) In order to solve the business problem we need to understand the business  first (what to Predict) later we have to divide variables in to dependent and independent, then we need to understand the dataset, data preperation , data modelling, evaluation, deployment.
# c) Dependent variables: Legendary ; Independent variables : Type1,Type2,Total,HP, Attack, Defence, Sp.Atk, Sp.Def, Speed, Generation

# In[3]:


# Importing the libraries
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns











# In[5]:


# Importing the dataset
csv1 = pd.read_csv("poke.csv")
csv1


# In[4]:


csv1.shape


# In[6]:


csv1.head()


# In[7]:


# Summary
csv1.describe()


# In[6]:


# Identify Outliers using Standard Deviation
mean=np.mean(csv1,axis=0)
std=np.std(csv1,axis=0)
std


# In[8]:


# Data types of each column in Dataset
csv1.dtypes


# In[9]:


# deleting the "ID" coloumn
del csv1['#']
csv1


# In[10]:


pd.pivot_table(csv1,index = ['Type 1','Type 2','Name'])


# In[11]:


csv1.head()


# In[12]:


# Checking for maximum ‘Hp’ for a Pokémon along with the ‘Name’
csv1[csv1['HP']==csv1['HP'].max()]['Name']


# In[13]:


# Checking for minimum ‘Hp’ for a Pokémon along with the ‘Name’
csv1[csv1['HP']==csv1['HP'].min()]['Name']


# In[14]:


dg = csv1.astype({"Type 1":'category',"Type 2":'category'})
dg.dtypes


# In[15]:


#  The Correlation Matrix of Dataset
corelations = csv1.corr()
corelations


# In[15]:


# corelation heat map
plt.figure(figsize=(10,8))
corelations = corelations * 100
sns.heatmap(corelations,annot = True,fmt = '.0f')


# In[16]:


# Checking  for Missing Values in  the Dataset
csv1.isnull()


# In[16]:


#Counting the number of missing values in each column
csv1.isnull().sum(axis=0)


# In[18]:


# Replacing Missing Values in ‘Type2’ with my name
csv1["Type 2"].fillna("suresh kumar", inplace = True)


# In[20]:


csv1.head()


# In[22]:


missing_col1 = csv1[['Total']]
mean = csv1[['Total']].mean()
k = missing_col1.fillna(mean)
s = round(k)
s.head()


# In[21]:


csv1['Total'] = s


# In[24]:


missing_col = csv1[['Sp. Atk']]
mean = csv1[['Sp. Atk']].mean()
g = missing_col.fillna(mean)
m = round(g)
m.head()


# In[25]:


csv1['Sp. Atk'] = m


# In[26]:


csv1.head()


# In[25]:


# Unique values in HP column
csv1['HP'].unique() 


# In[27]:


# Unique values in HP column
csv1['HP'].unique() 


# In[28]:


# Unique values in Attack column
csv1['Attack'].unique() 


# In[29]:


# Unique values in Sp. Atk column
csv1['Sp. Atk'].unique() 


# In[31]:


# Unique values in Sp. Def column
csv1['Sp. Def'].unique() 


# In[30]:


# Unique values in Legendary column
csv1['Legendary'].unique() 


# In[33]:


# Boxplot for variable HP
sns.boxplot(csv1['HP'])


# In[32]:


# Boxplot for variable Attack
sns.boxplot(csv1['Attack'])


# In[33]:


# Boxplot for variable Sp. Atk
sns.boxplot(csv1['Sp. Atk'])


# In[34]:


# Boxplot for variable Defense
sns.boxplot(csv1['Defense'])


# In[35]:


# Boxplot for variable Speed
sns.boxplot(csv1['Speed'])


# In[36]:


# Boxplot for variable Sp. Def
sns.boxplot(csv1['Sp. Def'])


# In[37]:


# Histogram for variable HP
plt.hist(csv1['HP'])


# In[38]:


# Histogram for variable Attack
plt.hist(csv1['Attack'])


# In[39]:


# Histogram for variable Sp. Atk
plt.hist(csv1['Sp. Atk'])


# In[40]:


# Histogram for variable Defense
plt.hist(csv1['Defense'])


# In[41]:


# Histogram for variable Speed
plt.hist(csv1['Speed'])


# In[42]:


# Histogram for variable Sp. Def
plt.hist(csv1['Sp. Def'])


# In[34]:


# SCatter plot between two variables
plt.scatter(csv1['Sp. Atk'],csv1['Sp. Def'])
plt.title('Scatter Plot')
plt.xlabel('Sp. Atk')
plt.ylabel('Sp. Def')
plt.show()


# In[35]:


#function to compare different stats between two Pokémon visually
def compare_pok(n1,n2,param):
    a = csv1[(csv1.Name == n1) | (csv1.Name == n2)]
    sns.factorplot(x='Name', y=param, data=a, kind='bar', size=5, aspect=1, palette = ["#0000ff","#FFB6C1"])


# In[36]:


csv1.head()


# In[37]:


compare_pok('Venusaur', 'Charmander','Total')


# In[38]:


##Plotting  Categorical plots using sea born
sns.catplot(data = csv1 )


# In[48]:


#Analysis from categorical plot


# In[49]:


# Finding the outliers
csv1.describe()


# In[39]:


# Dealing outliers Using Percentile Values
def detect_outlier(csv1):
    for i in csv1.describe().columns:
        Q1 = csv1.describe().at['25%',i]
        Q3 = csv1.describe().at['75%',i]
        IQR = Q3 - Q1
        LTV = Q1 - 1.5*IQR 
        UTV = Q3 + 1.5*IQR
        csv1[i] = csv1[i].mask(csv1[i]<LTV,LTV)
        csv1[i] = csv1[i].mask(csv1[i]>UTV,UTV)
    return csv1


# In[40]:


mask = detect_outlier(csv1)


# In[41]:


mask


# In[43]:


# Applying standar scalar function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(csv1[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]))


# In[44]:


print(scaler.transform(csv1[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]))


# In[ ]:


# pandas profiling 


# In[ ]:


#Inputs: Total,Hp,Attack,Defence,Sp. Atk, Sp. Def, speed, generation
#output: Legendary


# In[45]:


# Dividing  Data into X and Y
import pandas as pd

csv1 = pd.read_csv("poke.csv")
array = csv1.values
X = array[:,3:11]
Y = array[:,11]


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=7)


# In[47]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Initialize parameters
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model1 = LogisticRegression()


# In[48]:


# Fitting the model and Extracting the results
results1 = cross_val_score(model1, X, Y, cv=kfold)


# In[ ]:





# In[ ]:




