from sklearn.linear_model import LogisticRegression
X=[[1],[2],[3],[6],[7],[8]]
y=[0,0,0,1,1,1]
m=LogisticRegression()
m.fit(X,y)
print("Prediction:",m.predict([[5]]))
print("------------")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = [[9.2, 90, 88],
    [8.8, 85, 84],
    [8.5, 82, 80],
    [7.2, 72, 70],
    [6.8, 68, 65],
    [9.0, 91, 89],
    [7.0, 70, 68],
    [8.7, 86, 83],
    [6.5, 65, 60],
    [9.1, 92, 90]]
y = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)
new_sample = [[8.9, 88, 85]]
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
if prediction[0] == 1:
    print("Predicted Result: Placed")
else:
    print("Predicted Result: Not Placed")
print("-------------")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = [[8, 780, 5],
    [7, 760, 4],
    [6, 730, 5],
    [4, 620, 8],
    [3, 590, 9],
    [9, 810, 4],
    [5, 650, 7],
    [8, 770, 5],
    [4, 610, 8],
    [7, 750, 6]]
y = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)
new_sample = [[7, 760, 5]]
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
if prediction[0] == 1:
    print("Predicted Result: Loan Approved")
else:
    print("Predicted Result: Loan Rejected")
print("------------")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = [[39.1, 112, 94],
    [38.8, 109, 95],
    [37.0, 78, 99],
    [39.3, 115, 93],
    [36.8, 74, 98],
    [38.7, 110, 94],
    [37.1, 80, 98],
    [39.0, 113, 92],
    [36.9, 76, 99],
    [38.9, 111, 93]]
y = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)
new_sample = [[38.8, 110, 94]]
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
if prediction[0] == 1:
    print("Predicted Result: Disease Present")
else:
    print("Predicted Result: Disease Absent")
print("-------------")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = [[10, 95, 50],
    [9, 92, 48],
    [8, 90, 45],
    [4, 70, 25],
    [3, 65, 20],
    [11, 96, 52],
    [5, 75, 30],
    [8, 89, 44],
    [2, 60, 18],
    [7, 87, 40]]
y = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)
new_sample = [[8, 91, 46]]
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
if prediction[0] == 1:
    print("Predicted Result: Promoted")
else:
    print("Predicted Result: Not Promoted")
print("--------------")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = [[22, 3, 30],
    [25, 4, 35],
    [28, 5, 45],
    [32, 7, 60],
    [35, 8, 65],
    [40, 9, 75],
    [45, 10, 80],
    [30, 6, 55],
    [24, 4, 40],
    [38, 9, 70]]
y = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)
new_sample = [[34, 8, 68]]
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
if prediction[0] == 1:
    print("Predicted Result: Purchased")
else:
    print("Predicted Result: Not Purchased")
    
