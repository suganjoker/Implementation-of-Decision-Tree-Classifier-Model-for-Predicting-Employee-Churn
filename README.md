# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: sugavarathan l
RegisterNumber:  212221220051
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data Head:
![head](https://user-images.githubusercontent.com/93427923/169693675-2a2f8bd7-9a87-49dc-a58c-777969b5f353.png)

### Information:
![info](https://user-images.githubusercontent.com/93427923/169693680-b6183dca-cdfb-4dad-afef-3badcecd05f9.png)

### Null dataset:
![null](https://user-images.githubusercontent.com/93427923/169693714-10634ad2-5b16-4db4-8b72-3d7b3babd95f.png)

### Value_counts():
![left](https://user-images.githubusercontent.com/93427923/169693730-1efadbf5-4cec-4d2b-bbdd-5d29fcaddc36.png)

### Data Head:
![head2](https://user-images.githubusercontent.com/93427923/169693736-5f392e94-f043-40fa-a0ed-32e89ad2ddb0.png)

### x.head():
![xhead](https://user-images.githubusercontent.com/93427923/169693739-0365b04f-731b-404b-b914-ef3b5b57c3cf.png)

### Accuracy:
![accuracy](https://user-images.githubusercontent.com/93427923/169693745-cd8c6451-7622-4ef9-a65c-3d7e3bd661de.png)

### Data Prediction:
![predict](https://user-images.githubusercontent.com/93427923/169693750-5106819e-ba64-4653-ad7b-b0f06df09a72.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
