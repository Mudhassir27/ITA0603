from sklearn.mixture import GaussianMixture
import numpy as np
X=np.array([[1],[2],[3],[10],[11],[12]])
g=GaussianMixture(n_components=2)
g.fit(X)
print(g.predict(X))
print("------------------")

from sklearn.mixture import GaussianMixture
X = [[2, 65],
    [3, 70],
    [4, 75],
    [5, 80],
    [6, 85],
    [7, 88],
    [8, 90],
    [9, 93],
    [10, 95],
    [11, 98]]
model = GaussianMixture(n_components=2, random_state=42)
model.fit(X)
labels = model.predict(X)
new_sample = [[7, 89]]
new_label = model.predict(new_sample)
print("Student Performance Clustering")
for i in range(len(X)):
    print("Study Hours:", X[i][0], 
          "Attendance:", X[i][1], 
          "Cluster:", labels[i])
print()
print("New Sample: Study Hours = 7, Attendance = 89")
print("New Sample Cluster:", new_label[0])
print("------------------------------")

from sklearn.mixture import GaussianMixture
X = [[2, 20],
    [3, 25],
    [4, 30],
    [5, 40],
    [6, 50],
    [8, 65],
    [10, 75],
    [12, 82],
    [14, 90],
    [16, 96]]
model = GaussianMixture(n_components=2, random_state=42)
model.fit(X)
labels = model.predict(X)
new_sample = [[9, 70]]
new_label = model.predict(new_sample)
print("Customer Segmentation")
for i in range(len(X)):
    print("Income:", X[i][0],
          "Lakhs, Spending Score:", X[i][1],
          "Cluster:", labels[i])
print()
print("New Sample: Income = 9 Lakhs, Spending Score = 70")
print("New Sample Cluster:", new_label[0])
print("---------------------")

from sklearn.mixture import GaussianMixture
X = [[800, 30],
    [900, 35],
    [1000, 40],
    [1200, 50],
    [1400, 60],
    [1600, 75],
    [1800, 90],
    [2000, 105],
    [2200, 120],
    [2400, 135]]
model = GaussianMixture(n_components=2, random_state=42)
model.fit(X)
labels = model.predict(X)
new_sample = [[1700, 82]]
new_label = model.predict(new_sample)
print("House Size Clustering")
for i in range(len(X)):
    print("House Size:", X[i][0],
          "sq.ft, Price:", X[i][1],
          "Lakhs, Cluster:", labels[i])
print()
print("New Sample: House Size = 1700 sq.ft, Price = 82 Lakhs")
print("New Sample Cluster:", new_label[0])
print("---------------------")

from sklearn.mixture import GaussianMixture
X = [[1, 3.0],
    [2, 3.8],
    [3, 4.8],
    [4, 6.0],
    [5, 7.5],
    [6, 9.0],
    [7, 10.8],
    [8, 12.6],
    [9, 14.5],
    [10, 16.5]]
model = GaussianMixture(n_components=2, random_state=42)
model.fit(X)
labels = model.predict(X)
new_sample = [[7, 11.0]]
new_label = model.predict(new_sample)
print("Employee Salary Clustering")
for i in range(len(X)):
    print("Experience:", X[i][0],
          "Years, Salary:", X[i][1],
          "Lakhs, Cluster:", labels[i])
print()
print("New Sample: Experience = 7 years, Salary = 11 Lakhs")
print("New Sample Cluster:", new_label[0])
print("--------------------------")

from sklearn.mixture import GaussianMixture
X = [[40, 4.5],
    [50, 5.0],
    [60, 5.6],
    [70, 6.5],
    [80, 7.8],
    [90, 9.2],
    [100, 10.8],
    [110, 12.5],
    [120, 14.3],
    [130, 16.2]]
model = GaussianMixture(n_components=2, random_state=42)
model.fit(X)
labels = model.predict(X)
new_sample = [[95, 10.0]]
new_label = model.predict(new_sample)
print("Vehicle Speed Clustering")
for i in range(len(X)):
    print("Speed:", X[i][0],
          "km/h, Fuel Consumption:", X[i][1],
          "L/100 km, Cluster:", labels[i])
print()
print("New Sample: Speed = 95 km/h, Fuel Consumption = 10.0 L/100 km")
print("New Sample Cluster:", new_label[0])
