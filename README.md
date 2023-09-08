# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Dharini PV
RegisterNumber: 212222240024
*/
```
```python
import pandas as pd
data = pd.read_csv('/content/Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:, :-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/3478a9db-4a5f-454d-9323-a20cbe6fdba2)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/62b9bf10-be78-4528-a76d-4683cebd135a)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/fee8a989-352a-4482-9688-1d5eff073c01)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/c080814c-fee9-42de-a162-29b4ef6a03e8)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/fcc408c6-532a-4765-add7-f33a3417580f)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/aab38888-db3b-4f98-b74f-c912cc3d339f)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/2d49b6a5-148a-4e10-bf67-7b046ab0e2ff)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/c24e6530-bfd5-461d-a12b-751a59cd377b)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/36110905-23e9-4f82-828b-dd1e6b06d955)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/e6f297a3-24e7-4ef3-886f-7791037081bf)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/b567d7e3-fbd8-43d4-bbdf-3e70209d6df3)

![image](https://github.com/DHARINIPV/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119400845/6c6a2946-0e0f-497a-b4f2-4ea5cbbdb0e4)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
