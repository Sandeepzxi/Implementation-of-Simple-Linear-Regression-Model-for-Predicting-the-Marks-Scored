# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Store data in a structured format.
2.Use a Simple Linear Regression model to fit the training data.
3.Use the trained model to predict values for the test set.
4. Evaluate performance using metrics.
## Program:
Developed by: Sandeep S
RegisterNumber: 212223220092
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
```

## Output:
![image](https://github.com/user-attachments/assets/2540324f-75b2-42e3-b3d4-12d9075b185f)
## df.tail()
# Output:
![image](https://github.com/user-attachments/assets/5a38a2cd-b386-4b84-8c1b-5c357744143e)

```
x=df.iloc[:,:-1].values
x
```
# Output:
![image](https://github.com/user-attachments/assets/046d1625-8552-4e7f-a28f-099ab4adaba7)
```
y=df.iloc[:,1].values
y
```

# Output:
![image](https://github.com/user-attachments/assets/934e96be-8a52-4260-bf56-bec755a458a4)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
```

```
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
```
# Tail Values:
![image](https://github.com/user-attachments/assets/5339a5a2-e4a4-4ed0-a1a8-7b033621513f)

# X Values:
![image](https://github.com/user-attachments/assets/40a2e414-6de5-4c55-8404-e87d799f0197)

y_pred
# Output:
![image](https://github.com/user-attachments/assets/c6c587bf-7bcc-430f-827e-7ead6e103731)

y_test
# Output:
![image](https://github.com/user-attachments/assets/e4e7117f-5a2a-416a-8ee0-709df052a4ce)
```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)
```
# Output:
![image](https://github.com/user-attachments/assets/dff3aebd-8aa6-4176-ab29-9267dbaee0b4)
```

plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```
# Output:
```

plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
# Output
![image](https://github.com/user-attachments/assets/05962cf0-eaec-4622-9a94-25b4b89b287f)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
