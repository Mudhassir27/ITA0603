from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
X,y=load_iris(return_X_y=True)
m=GaussianNB()
m.fit(X,y)
print(m.predict([X[0]]))
print("--------------------")

from sklearn.naive_bayes import GaussianNB
X=[
[150,7.5,90],
[160,7.8,88],
[145,7.3,91],
[120,5.5,70],
[125,5.8,72],
[130,6.0,74],
[155,7.6,89],
[118,5.4,69],
[148,7.4,92],
[122,5.7,71]
]
y=[
"Apple",
"Apple",
"Apple",
"Orange",
"Orange",
"Orange",
"Apple",
"Orange",
"Apple",
"Orange"
]
model=GaussianNB()
model.fit(X,y)
new_sample=[[152,7.5,90]]
prediction=model.predict(new_sample)
print("Predicted Fruit:",prediction[0])
print("----------------------")

from sklearn.naive_bayes import GaussianNB
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
model=GaussianNB()
model.fit(X,y)
new_sample=[[8.7,86,84]]
prediction=model.predict(new_sample)
print("Predicted Placement:",prediction[0])
print("-----------------------")

from sklearn.naive_bayes import GaussianNB
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
model=GaussianNB()
model.fit(X,y)
new_sample=[[38.7,109,94]]
prediction=model.predict(new_sample)
print("Predicted Disease:",prediction[0])
print("-----------------------")

from sklearn.naive_bayes import GaussianNB
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
model=GaussianNB()
model.fit(X,y)
new_sample=[[7,90,44]]
prediction=model.predict(new_sample)
print("Predicted Promotion:",prediction[0])
print("-------------------------")

from sklearn.naive_bayes import GaussianNB
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
model=GaussianNB()
model.fit(X,y)
new_sample=[[135,118,60]]
prediction=model.predict(new_sample)
print("Predicted Vehicle Type:",prediction[0])
