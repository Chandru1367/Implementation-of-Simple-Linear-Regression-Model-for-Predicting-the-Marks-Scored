# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Chandru M
RegisterNumber: 24900224
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train, regressor.predict(x_train),color='blue') 
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train, regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE=',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE=',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
## Output:
![Screenshot 2024-10-18 135113](https://github.com/user-attachments/assets/90cee5f5-3a7a-4ceb-85d0-db8b489af38d)
![Screenshot 2024-10-18 135125](https://github.com/user-attachments/assets/94d06141-9824-4a7f-b925-bcbf0d1a45b5)
![Screenshot 2024-10-18 135134](https://github.com/user-attachments/assets/5eeda295-f515-4b41-aa89-95740d933e6f)
![Screenshot 2024-10-18 135244](https://github.com/user-attachments/assets/20a43fc6-b7cd-43b4-9abd-046477151806)
![Screenshot 2024-10-18 135253](https://github.com/user-attachments/assets/ef25205a-b00b-4956-a8b1-d142bec6f849)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
