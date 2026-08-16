from sklearn.linear_model import LinearRegression
X=[[4],[6],[8],[10]]
y=[10000,15000,20000,25000]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[12]]))
print("-------------------")

from sklearn.linear_model import LinearRegression
X = [
    [4, 64, 12],
    [4, 128, 16],
    [6, 128, 48],
    [6, 256, 50],
    [8, 128, 64],
    [8, 256, 64],
    [12, 256, 108],
    [12, 512, 108],
    [16, 512, 200],
    [16, 1024, 200]
]
y = [12000, 15000, 18000, 22000, 25000,
     30000, 38000, 45000, 60000, 75000]
model = LinearRegression()
model.fit(X, y)
new_mobile = [[8, 256, 108]]
prediction = model.predict(new_mobile)
print("Predicted Mobile Price: ₹", round(prediction[0], 2))
print("-------------------")

from sklearn.linear_model import LinearRegression
X = [
    [1.8, 4, 4000],
    [2.0, 4, 4500],
    [2.2, 6, 5000],
    [2.4, 6, 5000],
    [2.6, 8, 5500],
    [2.8, 8, 6000],
    [3.0, 12, 6000],
    [3.2, 12, 6500],
    [3.4, 16, 7000],
    [3.6, 16, 7000]
]
y = [10000, 12000, 17000, 21000, 26000,
     32000, 40000, 50000, 62000, 76000]
model = LinearRegression()
model.fit(X, y)
new_mobile = [[2.9, 8, 6000]]
prediction = model.predict(new_mobile)
print("Predicted Mobile Price: ₹", round(prediction[0], 2))
print("------------------")

from sklearn.linear_model import LinearRegression
X = [
    [6.1, 60, 4000],
    [6.2, 90, 4500],
    [6.4, 90, 5000],
    [6.5, 120, 5000],
    [6.6, 120, 5500],
    [6.7, 120, 6000],
    [6.8, 144, 6000],
    [6.9, 144, 6500],
    [7.0, 165, 7000],
    [7.1, 165, 7000]
]
y = [12000, 15000, 18000, 23000, 28000,
     34000, 42000, 52000, 65000, 78000]
model = LinearRegression()
model.fit(X, y)
new_mobile = [[6.7, 144, 6000]]
prediction = model.predict(new_mobile)
print("Predicted Mobile Price: ₹", round(prediction[0], 2))
print("--------------------")

from sklearn.linear_model import LinearRegression
X = [
    [4, 6, 64],
    [6, 8, 128],
    [8, 8, 128],
    [8, 8, 256],
    [12, 8, 256],
    [12, 8, 512],
    [16, 8, 512],
    [16, 8, 1024],
    [18, 8, 1024],
    [24, 8, 1024]
]
y = [14000, 19000, 26000, 32000, 42000,
     52000, 65000, 82000, 95000, 120000]
model = LinearRegression()
model.fit(X, y)
new_mobile = [[12, 8, 512]]
prediction = model.predict(new_mobile)
print("Predicted Mobile Price: ₹", round(prediction[0], 2))
print("-------------------")

from sklearn.linear_model import LinearRegression
X = [
    [250000, 4000, 48],
    [320000, 4500, 50],
    [420000, 5000, 64],
    [520000, 5000, 64],
    [620000, 5500, 108],
    [720000, 6000, 108],
    [820000, 6000, 200],
    [920000, 6500, 200],
    [1020000, 7000, 200],
    [1120000, 7000, 200]
]
y = [15000, 18000, 24000, 30000, 38000,
     46000, 58000, 70000, 85000, 100000]
model = LinearRegression()
model.fit(X, y)
new_mobile = [[750000, 6000, 108]]
prediction = model.predict(new_mobile)
print("Predicted Mobile Price: ₹", round(prediction[0], 2))
