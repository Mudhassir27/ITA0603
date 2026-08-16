from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
X,y=load_iris(return_X_y=True)
print("DT:",DecisionTreeClassifier().fit(X,y).score(X,y))
print("KNN:",KNeighborsClassifier().fit(X,y).score(X,y))
print("-------------------------")


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X=[
[9.2,90,88],
[8.8,85,82],
[8.5,80,79],
[7.2,72,70],
[6.8,68,65],
[9.0,92,90],
[7.5,75,74],
[8.6,84,81],
[6.5,60,58],
[8.9,88,86]
]
y=[
"Placed",
"Placed",
"Placed",
"Not Placed",
"Not Placed",
"Placed",
"Not Placed",
"Placed",
"Not Placed",
"Placed"
]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
dt=DecisionTreeClassifier(random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
dt.fit(X_train,y_train)
knn.fit(X_train,y_train)
dt_pred=dt.predict(X_test)
knn_pred=knn.predict(X_test)
print("Decision Tree Accuracy:",accuracy_score(y_test,dt_pred))
print("KNN Accuracy:",accuracy_score(y_test,knn_pred))
new_sample=[[8.7,86,84]]
print("Decision Tree Prediction:",dt.predict(new_sample)[0])
print("KNN Prediction:",knn.predict(new_sample)[0])
print("---------------------------")

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X=[
[39.0,110,94],
[38.5,108,95],
[37.0,78,99],
[39.2,115,93],
[36.8,75,98],
[38.8,112,94],
[37.2,80,98],
[39.1,114,92],
[36.7,74,99],
[38.9,111,93]
]
y=[
"Positive",
"Positive",
"Negative",
"Positive",
"Negative",
"Positive",
"Negative",
"Positive",
"Negative",
"Positive"
]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
dt=DecisionTreeClassifier(random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
dt.fit(X_train,y_train)
knn.fit(X_train,y_train)
dt_pred=dt.predict(X_test)
knn_pred=knn.predict(X_test)
print("Decision Tree Accuracy:",accuracy_score(y_test,dt_pred))
print("KNN Accuracy:",accuracy_score(y_test,knn_pred))
new_sample=[[38.7,109,94]]
print("Decision Tree Prediction:",dt.predict(new_sample)[0])
print("KNN Prediction:",knn.predict(new_sample)[0])
print("----------------------------")

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X=[
[39.0,110,94],
[38.5,108,95],
[37.0,78,99],
[39.2,115,93],
[36.8,75,98],
[38.8,112,94],
[37.2,80,98],
[39.1,114,92],
[36.7,74,99],
[38.9,111,93]
]
y=[
"Positive",
"Positive",
"Negative",
"Positive",
"Negative",
"Positive",
"Negative",
"Positive",
"Negative",
"Positive"
]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
dt=DecisionTreeClassifier(random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
dt.fit(X_train,y_train)
knn.fit(X_train,y_train)
dt_pred=dt.predict(X_test)
knn_pred=knn.predict(X_test)
print("Decision Tree Accuracy:",accuracy_score(y_test,dt_pred))
print("KNN Accuracy:",accuracy_score(y_test,knn_pred))
new_sample=[[38.7,109,94]]
print("Decision Tree Prediction:",dt.predict(new_sample)[0])
print("KNN Prediction:",knn.predict(new_sample)[0])
print("--------------------------")

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X=[
[10,95,50],
[8,90,45],
[7,88,40],
[3,65,20],
[2,60,18],
[9,93,48],
[4,70,25],
[8,89,42],
[2,58,15],
[6,85,38]
]
y=[
"Promoted",
"Promoted",
"Promoted",
"Not Promoted",
"Not Promoted",
"Promoted",
"Not Promoted",
"Promoted",
"Not Promoted",
"Promoted"
]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
dt=DecisionTreeClassifier(random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
dt.fit(X_train,y_train)
knn.fit(X_train,y_train)
dt_pred=dt.predict(X_test)
knn_pred=knn.predict(X_test)
print("Decision Tree Accuracy:",accuracy_score(y_test,dt_pred))
print("KNN Accuracy:",accuracy_score(y_test,knn_pred))
new_sample=[[7,90,44]]
print("Decision Tree Prediction:",dt.predict(new_sample)[0])
print("KNN Prediction:",knn.predict(new_sample)[0])
print("---------------------------")

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X=[
[100,95,70],
[125,110,65],
[150,125,55],
[800,850,22],
[1000,950,20],
[1200,1050,18],
[110,100,68],
[900,900,21],
[140,120,58],
[1300,1100,17]
]
y=[
"Bike",
"Bike",
"Bike",
"Car",
"Car",
"Car",
"Bike",
"Car",
"Bike",
"Car"
]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
dt=DecisionTreeClassifier(random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
dt.fit(X_train,y_train)
knn.fit(X_train,y_train)
dt_pred=dt.predict(X_test)
knn_pred=knn.predict(X_test)
print("Decision Tree Accuracy:",accuracy_score(y_test,dt_pred))
print("KNN Accuracy:",accuracy_score(y_test,knn_pred))
new_sample=[[135,118,60]]
print("Decision Tree Prediction:",dt.predict(new_sample)[0])
print("KNN Prediction:",knn.predict(new_sample)[0])
