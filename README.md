# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Abinav Aaditya
RegisterNumber: 212224040008

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]]) 
*/
```

## Output:
### Data Head:

<img width="1391" height="257" alt="m81" src="https://github.com/user-attachments/assets/ba34f79b-b830-405e-86ff-c0b3f0a4b75f" />


### Dataset info :

<img width="498" height="362" alt="m82" src="https://github.com/user-attachments/assets/fd5dc56b-6037-4d7d-b63f-09efdcecc8bc" />


### Null Dataset:

<img width="312" height="255" alt="m83" src="https://github.com/user-attachments/assets/d86b833c-2c67-45a4-b1c5-b0a598cb27c6" />


### Values count in left column:

<img width="312" height="97" alt="m84" src="https://github.com/user-attachments/assets/ea60c8cf-d0b0-41ea-a40c-468348bacc07" />


### Dataset transformed head:

<img width="1371" height="242" alt="m85" src="https://github.com/user-attachments/assets/4b0be067-0419-49a5-b56c-e7e3c6fe3e63" />


### x.head:

<img width="1228" height="232" alt="m86" src="https://github.com/user-attachments/assets/1838ad18-8c1f-424b-9e44-84ac95c746af" />


### Accuracy:

<img width="257" height="52" alt="m87" src="https://github.com/user-attachments/assets/11a8809c-50d3-432e-b503-8f0691b7fdc5" />


### Data prediction:

<img width="1372" height="97" alt="m88" src="https://github.com/user-attachments/assets/165c7d73-3bb5-48bd-8275-0cd4b3c3b4cc" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
