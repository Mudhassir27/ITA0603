from sklearn.linear_model import LinearRegression
X=[[1],[2],[3],[4]]
y=[100,200,300,400]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[5]]))
print("----------------------")

from sklearn.linear_model import LinearRegression
X = [
    [20, 2],
    [25, 2],
    [30, 3],
    [35, 3],
    [40, 4],
    [45, 4],
    [50, 5],
    [55, 5],
    [60, 6],
    [65, 6]
]
y = [250, 290, 340, 390, 450, 510, 580, 640, 710, 780]
model = LinearRegression()
model.fit(X, y)
new_data = [[48, 5]]
prediction = model.predict(new_data)
print("Predicted Monthly Sales:", round(prediction[0], 2), "Units")
print("------------------------")

from sklearn.linear_model import LinearRegression
X = [
    [1000, 120],
    [1200, 150],
    [1400, 180],
    [1600, 220],
    [1800, 260],
    [2000, 300],
    [2200, 340],
    [2400, 380],
    [2600, 420],
    [2800, 460]
]
y = [8, 10, 12, 15, 18, 22, 26, 31, 36, 42]
model = LinearRegression()
model.fit(X, y)
new_data = [[2100, 320]]
prediction = model.predict(new_data)
print("Predicted Monthly Sales:", round(prediction[0], 2), "Lakhs")
print("------------------------")

from sklearn.linear_model import LinearRegression
X = [
    [1000, 5000],
    [1500, 7000],
    [2000, 9000],
    [2500, 11000],
    [3000, 13000],
    [3500, 15000],
    [4000, 17000],
    [4500, 19000],
    [5000, 21000],
    [5500, 23000]
]
y = [120, 180, 250, 320, 400, 490, 590, 700, 820, 950]
model = LinearRegression()
model.fit(X, y)
new_data = [[4200, 18000]]
prediction = model.predict(new_data)
print("Predicted Sales:", round(prediction[0], 2), "Orders")
print("--------------------------")

from sklearn.linear_model import LinearRegression
X = [
    [5, 150],
    [8, 180],
    [10, 220],
    [12, 260],
    [15, 300],
    [18, 340],
    [20, 380],
    [22, 420],
    [25, 460],
    [28, 500]
]
y = [220, 260, 310, 360, 420, 490, 570, 650, 740, 840]
model = LinearRegression()
model.fit(X, y)
new_data = [[18, 350]]
prediction = model.predict(new_data)
print("Predicted Weekly Sales:", round(prediction[0], 2), "Thousand")
print("-------------------------")

from sklearn.linear_model import LinearRegression
X = [
    [30, 20],
    [35, 25],
    [40, 30],
    [45, 35],
    [50, 40],
    [55, 45],
    [60, 50],
    [65, 55],
    [70, 60],
    [75, 65]
]
y = [8, 10, 13, 17, 22, 28, 35, 43, 52, 62]
model = LinearRegression()
model.fit(X, y)
new_data = [[58, 48]]
prediction = model.predict(new_data)
print("Predicted Cars Sold:", round(prediction[0], 2))
