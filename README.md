# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1..Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:

```python
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: CHANDRU M
RegisterNumber:  24900224
```

```python
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
```

## Output:
![Screenshot 2024-10-18 143514](https://github.com/user-attachments/assets/6a82fd2f-6759-42e4-8bd9-dab143d2f528)
![Screenshot 2024-10-18 143502](https://github.com/user-attachments/assets/7b77ce6d-2ebd-4456-a179-ba2cd6b3b9a4)
![Screenshot 2024-10-18 143455](https://github.com/user-attachments/assets/60747aa6-d508-4729-9389-fd2a288d9826)
![Screenshot 2024-10-18 143441](https://github.com/user-attachments/assets/449efaeb-7b9f-43d9-af14-3b71f341961d)
![Screenshot 2024-10-18 143433](https://github.com/user-attachments/assets/e921bb28-68a0-4d0d-8762-f239405ac4d9)
![Screenshot 2024-10-18 143315](https://github.com/user-attachments/assets/d0b5e977-327e-44d2-b302-d670cb44b1ff)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
