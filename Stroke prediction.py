#!/usr/bin/env python
# coding: utf-8

# This project deals with the prediction of stroke based on lifestyle and basic bioparameters of the patient. The csv dataset is obtained from kaggle which has to be cleaned and processed before training.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("healthcare-dataset-stroke-data.csv")
data.head()


# The Id column is not needed for the prediction of stroke. Hence we can remove that column from the dataset.

# In[3]:


data=data.drop(['id'],axis=1)
data.head()


# In[4]:


data.info()


# We can see that there are 5110 rows with 11 columns, out of which 10 are input features and 1 is output target variable. There are some empty values associated with that of BMI column.

# In[5]:


data.isnull().sum()


# In[6]:


data.describe()


# ## visualizations

# In[7]:


age_data = data['age']

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(age_data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Show the plot
plt.grid(True)
plt.show()


# In[8]:


hypertension_data = data['hypertension']
heart_disease_data = data['heart_disease']

# Calculate the counts of hypertension and heart disease
hypertension_counts = hypertension_data.value_counts()
heart_disease_counts = heart_disease_data.value_counts()

# Create a bar chart
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Bar chart for Hypertension
axes[0].bar(hypertension_counts.index, hypertension_counts.values, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Hypertension Distribution')
axes[0].set_xlabel('Hypertension')
axes[0].set_ylabel('Frequency')

# Bar chart for Heart Disease
axes[1].bar(heart_disease_counts.index, heart_disease_counts.values, color='salmon', edgecolor='black', alpha=0.7)
axes[1].set_title('Heart Disease Distribution')
axes[1].set_xlabel('Heart Disease')
axes[1].set_ylabel('Frequency')

# Show the plots
plt.tight_layout()
plt.show()


# In[9]:


avg_glucose_level = data['avg_glucose_level']
bmi = data['bmi']
stroke = data['stroke'] 

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(avg_glucose_level, bmi, c=stroke, cmap='viridis', alpha=0.6)

# Add labels and title
plt.title('Average Glucose Level vs. BMI')
plt.xlabel('Average Glucose Level')
plt.ylabel('BMI')

# Add color bar legend for stroke presence or absence
cb = plt.colorbar()
cb.set_label('Stroke')

# Show the plot
plt.grid(True)
plt.show()


# Also we can observe that maximum value of BMI recorded is 97.6 which is physically impossible. It might be an observational error or they might have recorded the percentile value instead of the original value. Hence it would be appropriate to change those values to the logical range (0-50).
# Also we are filling the empty values based on the forward values using ffill method.

# In[10]:


#the maximum possible value for BMI would be 50, so the other values would be observational error. Hence it would be logical to convert them to the maximum value possible.
for x in data.index:
    if data.loc[x,'bmi']>50:
        data.loc[x,'bmi']=50
data['bmi']
data['bmi']=data['bmi'].fillna(method='ffill')
data.isnull().sum()


# Next we need to encode the string data types to corresponding categorical class numbers.

# In[11]:


data_categorical=data[['gender','ever_married','work_type','Residence_type','smoking_status']]
data_numerical=data[['age','hypertension','heart_disease','avg_glucose_level','bmi','stroke']]
print(data_categorical['gender'].unique())
print(data_categorical['ever_married'].unique())
print(data_categorical['work_type'].unique())
print(data_categorical['Residence_type'].unique())
print(data_categorical['smoking_status'].unique())


# In[12]:


df_categorical = pd.get_dummies(data_categorical, drop_first=True)
data=pd.concat([df_categorical,data_numerical],axis=1)
data.head()


# In[13]:


data.drop_duplicates() 


# ##  identify the outliers.  

# In[14]:


sns.boxplot(data['age'])
plt.show()


# In[15]:


sns.boxplot(data['avg_glucose_level'])
np.where(data['avg_glucose_level']>170)
lis=np.where(data['avg_glucose_level']>=170)
plt.show()


# In[16]:


sns.boxplot(data['bmi'])
plt.show()


# In[17]:


from scipy import stats
import numpy as np
 
z1 = np.abs(stats.zscore(data['age']))
z2 = np.abs(stats.zscore(data['avg_glucose_level']))
z3 = np.abs(stats.zscore(data['bmi']))
print('age'+str(np.where(z1>3)))
print('avg_glucose_level'+str(np.where(z2>3)))
print('bmi'+str(np.where(z3>3)))


# In[18]:


Q1 = np.percentile(data['age'], 25,
                   method = 'midpoint')
 
Q3 = np.percentile(data['age'], 75,
                   method = 'midpoint')
IQR = Q3 - Q1
upper = data['age'] >= (Q3+1.5*IQR)
#print("Upper bound:",upper)
print(np.where(upper))
 
# Below Lower bound
lower = data['age'] <= (Q1-1.5*IQR)
#print("Lower bound:", lower)
print(np.where(lower))


# In[19]:


Q1 = np.percentile(data['avg_glucose_level'], 25,
                   method = 'midpoint')
 
Q3 = np.percentile(data['avg_glucose_level'], 75,
                   method = 'midpoint')
IQR = Q3 - Q1

upper = data['avg_glucose_level'] >= (Q3+1.5*IQR)
print(np.where(upper))
 

lower = data['avg_glucose_level'] <= (Q1-1.5*IQR)
print(np.where(lower))


# In[20]:


Q1 = np.percentile(data['bmi'], 25,
                   method = 'midpoint')
 
Q3 = np.percentile(data['bmi'], 75,
                   method = 'midpoint')
IQR = Q3 - Q1

upper = data['bmi'] >= (Q3+1.5*IQR)
print(np.where(upper))
 
lower = data['bmi'] <= (Q1-1.5*IQR)
print(np.where(lower))


# As you can see there are certain values of average glucose levels and age that are beyond the 75 percentile range, hence detected as outliers by the IQR and bo plot. But in our case, they aren't outliers because stroke occurences may depend on age and high glucose levels which are considered as some of the risk factors for the disease. Hence we must not remove them. 

# I have planned to train the data using SVM, which is a distance based algorithm. Hence it would be appropriate to keep all the data in the same scale (0-1) so that the algorithm can work effectively. The age, average glucose level and bmi are in different ranges and have to be normalized to the range of 0-1.

# In[21]:


for column in data.columns:
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min()) 
data.head()


# In[22]:


data.info()


# In[23]:


data.describe()


# The data has been cleaned and processed. Now its good to go for training. The y (label) is the last column (stroke) and the x (data) are the columns other than stroke. Then the data is split into training and testing by using the train_test_split from sklearn.

# ## Training ML Model  

# In[24]:


y=data['stroke']
data=data.drop(['stroke'],axis=1)
X=data


# In[25]:


from sklearn.model_selection import train_test_split

X_train,X_test ,y_train, y_test = train_test_split(X,y,
                                                   test_size=0.20,
                                                   random_state=0,
                                                  shuffle = True,
                                                  stratify = y)

print('training data shape is :{}.'.format(X_train.shape))
print('training label shape is :{}.'.format(y_train.shape))
print('testing data shape is :{}.'.format(X_test.shape))
print('testing label shape is :{}.'.format(y_test.shape))


#  SVM is known to be a powerful algorithm especially in data that have non-linear relationships.

# In[26]:


from sklearn.svm import SVC
# Building a Support Vector Machine on train data
svc_model = SVC(kernel='poly',gamma=8)
svc_model.fit(X_train, y_train)


# In[27]:


from sklearn.metrics import accuracy_score, confusion_matrix
predictions= svc_model .predict(X_train)
percentage=svc_model.score(X_train,y_train)
res=confusion_matrix(y_train,predictions)
print("Training confusion matrix")
print(res)
predictions= svc_model .predict(X_test)
percentage=svc_model.score(X_test,y_test)
res=confusion_matrix(y_test,predictions)
print("validation confusion matrix")
print(res)
# check the accuracy on the training set
print('training accuracy = '+str(svc_model.score(X_train, y_train)*100))
print('testing accuracy = '+str(svc_model.score(X_test, y_test)*100))


# In[28]:


y_train.value_counts()


# In[29]:


y_test.value_counts()


# The algorithm has produced accuracy more than 90%, but the model is not good enough. Because the data suffers from class imbalance. Hence treating that condition would improve the metrics even more. Hence we are now going to resample the data.

# In[30]:


data=pd.read_csv("healthcare-dataset-stroke-data.csv")
data=data.drop(['id'],axis=1)
for x in data.index:
    if data.loc[x,'bmi']>50:
        data.loc[x,'bmi']=50
data['bmi']
data['bmi']=data['bmi'].fillna(method='ffill')
data.isnull().sum()
data_categorical=data[['gender','ever_married','work_type','Residence_type','smoking_status']]
data_numerical=data[['age','hypertension','heart_disease','avg_glucose_level','bmi','stroke']]
df_categorical = pd.get_dummies(data_categorical, drop_first=True)
data=pd.concat([df_categorical,data_numerical],axis=1)
for column in data.columns:
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min()) 

data.head()


# In[31]:


print(data['stroke'].value_counts())
df_class_0 = data[data['stroke'] == 0]
df_class_1 = data[data['stroke'] == 1]


# As you can see, there are only 249 entries for stroke 1 class. Hence we are going to upsample the minority class.

# In[32]:


df_class_1_over = df_class_1.sample(4861, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)


# In[33]:


df_test_over.info()


# In[34]:


y1=df_test_over['stroke']
df_test_over=df_test_over.drop(['stroke'],axis=1)
X1=df_test_over


# In[35]:


from sklearn.model_selection import train_test_split

X1_train,X1_test ,y1_train, y1_test = train_test_split(X1,y1,
                                                   test_size=0.2,
                                                   random_state=0,
                                                  shuffle = True,
                                                  stratify = y1)

print('training data shape is :{}.'.format(X1_train.shape))
print('training label shape is :{}.'.format(y1_train.shape))
print('testing data shape is :{}.'.format(X1_test.shape))
print('testing label shape is :{}.'.format(y1_test.shape))


# In[36]:


from sklearn.svm import SVC
svc_model = SVC(kernel='rbf',gamma=8)
svc_model.fit(X1_train, y1_train)


# In[37]:


from sklearn.metrics import accuracy_score, confusion_matrix
predictions= svc_model .predict(X1_train)
percentage=svc_model.score(X1_train,y1_train)
res=confusion_matrix(y1_train,predictions)
print("Training confusion matrix")
print(res)
predictions= svc_model .predict(X1_test)
percentage=svc_model.score(X1_test,y1_test)
res=confusion_matrix(y1_test,predictions)
print("validation confusion matrix")
print(res)
# check the accuracy on the training set
print('training accuracy = '+str(svc_model.score(X1_train, y1_train)*100))
print('testing accuracy = '+str(svc_model.score(X1_test, y1_test)*100))


# Even though the accuracy has decreased, the number of false negatives have greatly decreased 
# False negatives for training before sampling= 141 
# False negatives for testing before sampling= 46
# False negatives for training after sampling= 61
# False negatives for testing after sampling= 18
# We can train with poly kernel to further more reduce the false positives and negatives, but that takes a lot of time!!
