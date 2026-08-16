from sklearn.naive_bayes import GaussianNB
X=[[25],[35],[45],[55]]
y=['No','No','Yes','Yes']
m=GaussianNB()
m.fit(X,y)
print(m.predict([[50]]))
print("-----------------------")

from sklearn.naive_bayes import GaussianNB
X = [
    [12, 820, 5],
    [10, 790, 6],
    [9, 760, 7],
    [7, 700, 8],
    [6, 680, 9],
    [5, 650, 10],
    [4, 620, 12],
    [8, 730, 8],
    [3, 600, 13],
    [11, 810, 5]
]
y = [
    "Approved", "Approved", "Approved", "Approved", "Rejected",
    "Rejected", "Rejected", "Approved", "Rejected", "Approved"
]
model = GaussianNB()
model.fit(X, y)
new_data = [[9, 770, 6]]
prediction = model.predict(new_data)
print("Predicted Loan Status:", prediction[0])
print("------------------------")

from sklearn.naive_bayes import GaussianNB
X = [
    [90000, 850, 8000],
    [80000, 820, 9000],
    [70000, 780, 12000],
    [60000, 740, 15000],
    [50000, 690, 18000],
    [45000, 670, 20000],
    [40000, 640, 23000],
    [75000, 800, 10000],
    [35000, 620, 25000],
    [95000, 870, 7000]
]
y = [
    "Approved", "Approved", "Approved", "Approved", "Rejected",
    "Rejected", "Rejected", "Approved", "Rejected", "Approved"
]
model = GaussianNB()
model.fit(X, y)
new_data = [[72000, 790, 11000]]
prediction = model.predict(new_data)
print("Predicted Loan Status:", prediction[0])
print("-----------------------")

from sklearn.naive_bayes import GaussianNB
X = [
    [20, 10, 1],
    [18, 9, 1],
    [15, 8, 2],
    [12, 6, 2],
    [10, 5, 3],
    [8, 4, 3],
    [6, 3, 4],
    [16, 8, 2],
    [5, 2, 5],
    [22, 12, 1]
]
y = [
    "Approved", "Approved", "Approved", "Approved", "Rejected",
    "Rejected", "Rejected", "Approved", "Rejected", "Approved"
]
model = GaussianNB()
model.fit(X, y)
new_data = [[17, 8, 2]]
prediction = model.predict(new_data)
print("Predicted Loan Status:", prediction[0])
print("--------------------------")

from sklearn.naive_bayes import GaussianNB
X = [
    [15, 92, 6],
    [13, 89, 7],
    [11, 85, 8],
    [9, 80, 9],
    [7, 72, 10],
    [6, 68, 11],
    [5, 65, 12],
    [12, 87, 7],
    [4, 60, 13],
    [16, 94, 5]
]
y = [
    "Approved", "Approved", "Approved", "Approved", "Rejected",
    "Rejected", "Rejected", "Approved", "Rejected", "Approved"
]
model = GaussianNB()
model.fit(X, y)
new_data = [[12, 88, 7]]
prediction = model.predict(new_data)
print("Predicted Loan Status:", prediction[0])
print("----------------------------")

from sklearn.naive_bayes import GaussianNB
X = [
    [14, 3, 830],
    [12, 2.5, 810],
    [10, 2, 780],
    [8, 1.8, 740],
    [7, 1.5, 690],
    [6, 1.2, 660],
    [5, 1.0, 630],
    [11, 2.2, 790],
    [4, 0.8, 600],
    [15, 3.5, 850]
]
y = [
    "Approved", "Approved", "Approved", "Approved", "Rejected",
    "Rejected", "Rejected", "Approved", "Rejected", "Approved"
]
model = GaussianNB()
model.fit(X, y)
new_data = [[11, 2.4, 800]]
prediction = model.predict(new_data)
print("Predicted Loan Status:", prediction[0])
