# Implementation-of-Multivariate-Linear-Regression-Model-for-Sales-Prediction

## AIM:
To write a program to implement the multivariate linear regression model for sales prediction.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
### Step 1: 
Import the required standard libraries such as pandas, matplotlib.pyplot for the
implementation of the multivariate linear regression model for sales prediction.
### Step 2: 
Upload the dataset and check for any null value in the values provided using the
.isnull() function.
### Step 3: 
Declare X and Y variables with respect to the dataset values given by the
'Advertising.csv' file.
### Step 4: 
Predict the values by assigning a variable called y_pred.
### Step 5: 
Using the Mean Square Error (MSE) find the required straight line that fits the data.
### Step 6: 
Print the Mean Square Error(MSE) and R square error.
### Step 7:
 End of the program

## Program:
```
/*
Program to implement the multivariate linear regression model for sales prediction.
Developed by: prem chand T
RegisterNumber:  212219040113
*/

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Advertising.csv")
df.head()
df.describe()
df.isnull().sum()
df.shape
X = df[["TV", "Radio", "Newspaper"]]
X
Y = df["Sales"]
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state
= 101)
from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(X_train,Y_train)
Y_pred = l.predict(X_test)
X_test
print("Regression slope: ",l.coef_[0])
print("Regression Intercept: ",l.intercept_)
Y_pred
from sklearn import metrics
MSE = metrics.mean_squared_error(Y_test,Y_pred)
print("MSE is {}".format(MSE))
r2 = metrics.r2_score(Y_test,Y_pred)
print("R squared error is {}".format(r2))
l.predict([[150.3,240.5,234.5]])
```

## Output:
![multivariate linear regression model for sales prediction](sam.png)


## Result:
Thus the program to implement the multivariate linear regression model for sales prediction is written and verified using python programming.
