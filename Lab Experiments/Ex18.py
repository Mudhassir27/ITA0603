from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
X,y=load_iris(return_X_y=True)
m=Perceptron()
m.fit(X,y)
print(m.predict([X[0]]))
print("-------------------")

from sklearn.linear_model import Perceptron
X = [
    [150, 7.5, 90],
    [160, 7.8, 88],
    [145, 7.3, 91],
    [120, 5.5, 70],
    [125, 5.8, 72],
    [130, 6.0, 74],
    [155, 7.6, 89],
    [118, 5.4, 69],
    [148, 7.4, 92],
    [122, 5.7, 71]
]
y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X, y)
new_fruit = [[152, 7.5, 90]]
prediction = model.predict(new_fruit)
if prediction[0] == 1:
    print("Predicted Fruit: Apple")
else:
    print("Predicted Fruit: Orange")
print("-----------------------")

from sklearn.linear_model import Perceptron
X = [
    [9.3, 92, 90],
    [8.9, 87, 85],
    [8.6, 84, 82],
    [7.3, 74, 72],
    [6.9, 69, 67],
    [9.1, 91, 89],
    [7.4, 75, 73],
    [8.7, 85, 83],
    [6.6, 64, 60],
    [8.8, 88, 86]
]
y = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X, y)
new_student = [[8.8, 86, 84]]
prediction = model.predict(new_student)
if prediction[0] == 1:
    print("Predicted Result: Placed")
else:
    print("Predicted Result: Not Placed")
print("-----------------------")

from sklearn.linear_model import Perceptron
X = [
    [39.2, 112, 93],
    [38.8, 110, 94],
    [37.0, 78, 99],
    [39.1, 114, 92],
    [36.8, 76, 98],
    [38.9, 111, 94],
    [37.2, 80, 99],
    [39.3, 115, 92],
    [36.7, 74, 98],
    [38.7, 109, 95]
]
y = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X, y)
new_patient = [[38.9, 110, 94]]
prediction = model.predict(new_patient)
if prediction[0] == 1:
    print("Predicted Disease Status: Positive")
else:
    print("Predicted Disease Status: Negative")
print("-----------------------")

from sklearn.linear_model import Perceptron
X = [
    [10, 95, 50],
    [9, 92, 48],
    [8, 90, 45],
    [4, 72, 28],
    [3, 68, 24],
    [11, 96, 52],
    [5, 75, 30],
    [8, 89, 44],
    [2, 60, 18],
    [7, 87, 40]
]
y = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X, y)
new_employee = [[8, 90, 46]]
prediction = model.predict(new_employee)
if prediction[0] == 1:
    print("Predicted Promotion: Promoted")
else:
    print("Predicted Promotion: Not Promoted")
print("-----------------------")

from sklearn.linear_model import Perceptron
X = [
    [100, 95, 70],
    [125, 110, 65],
    [150, 125, 55],
    [800, 850, 22],
    [1000, 950, 20],
    [1200, 1050, 18],
    [110, 100, 68],
    [900, 900, 21],
    [140, 120, 58],
    [1300, 1100, 17]
]
y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X, y)
new_vehicle = [[135, 118, 60]]
prediction = model.predict(new_vehicle)
if prediction[0] == 1:
    print("Predicted Vehicle Type: Bike")
else:
    print("Predicted Vehicle Type: Car")
