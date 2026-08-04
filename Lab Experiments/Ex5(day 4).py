from sklearn.neighbors import KNeighborsClassifier
X=[[1],[2],[3],[6],[7],[8]]
y=[0,0,0,1,1,1]
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)
print("Prediction:",knn.predict([[5]]))
print("----------")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
X = [
    [9.2, 90, 88],
    [8.8, 85, 82],
    [8.5, 80, 79],
    [7.2, 72, 70],
    [6.8, 68, 65],
    [9.0, 92, 90],
    [7.5, 75, 74],
    [8.6, 84, 81],
    [6.5, 60, 58],
    [8.9, 88, 86]
]
y = [
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)
new_sample = [[8.7, 86, 84]]
new_sample_scaled = scaler.transform(new_sample)
prediction = knn.predict(new_sample_scaled)
print("Predicted Result:", prediction[0])
print("----------")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
X = [
    [8, 780, 5],
    [7, 760, 4],
    [6, 720, 6],
    [4, 620, 8],
    [3, 580, 10],
    [9, 800, 4],
    [5, 650, 7],
    [8, 770, 5],
    [4, 600, 9],
    [7, 740, 6]
]
y = [
    "Yes",
    "Yes",
    "Yes",
    "No",
    "No",
    "Yes",
    "No",
    "Yes",
    "No",
    "Yes"
]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)
new_sample = [[7, 750, 5]]
new_sample_scaled = scaler.transform(new_sample)
prediction = knn.predict(new_sample_scaled)
print("Predicted Result:", prediction[0])
print("-----------")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
X = [
    [39.0, 110, 94],
    [38.5, 108, 95],
    [37.0, 78, 99],
    [39.2, 115, 93],
    [36.8, 75, 98],
    [38.8, 112, 94],
    [37.2, 80, 98],
    [39.1, 114, 92],
    [36.7, 74, 99],
    [38.9, 111, 93]
]
y = [
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)
new_sample = [[38.7, 109, 94]]
new_sample_scaled = scaler.transform(new_sample)
prediction = knn.predict(new_sample_scaled)
print("Predicted Result:", prediction[0])
print("------------")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
X = [
    [10, 95, 50],
    [8, 90, 45],
    [7, 88, 40],
    [3, 65, 20],
    [2, 60, 18],
    [9, 93, 48],
    [4, 70, 25],
    [8, 89, 42],
    [2, 58, 15],
    [6, 85, 38]
]
y = [
    "Yes",
    "Yes",
    "Yes",
    "No",
    "No",
    "Yes",
    "No",
    "Yes",
    "No",
    "Yes"
]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)
new_sample = [[7, 90, 44]]
new_sample_scaled = scaler.transform(new_sample)
prediction = knn.predict(new_sample_scaled)
print("Predicted Result:", prediction[0])
print("-----------")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
X = [
    [150, 7.5, 90],
    [160, 7.8, 88],
    [140, 7.2, 91],
    [120, 5.5, 70],
    [125, 5.8, 72],
    [130, 6.0, 74],
    [155, 7.6, 89],
    [118, 5.4, 69],
    [148, 7.4, 92],
    [122, 5.7, 71]
]
y = [
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)
new_sample = [[152, 7.5, 90]]
new_sample_scaled = scaler.transform(new_sample)
prediction = knn.predict(new_sample_scaled)
print("Predicted Result:", prediction[0])
