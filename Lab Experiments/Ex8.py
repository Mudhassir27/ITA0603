from sklearn.linear_model import LinearRegression
X=[[1],[2],[3],[4],[5]]
y=[2,4,6,8,10]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[6]]))
print("-----------------")

from sklearn.linear_model import LinearRegression
X = [[800, 2, 10],
    [1000, 2, 8],
    [1200, 3, 6],
    [1400, 3, 5],
    [1600, 4, 4],
    [1800, 4, 3],
    [2000, 5, 2],
    [2200, 5, 1],
    [2400, 6, 1],
    [2600, 6, 0]]
y = [35, 45, 55, 65, 75, 85, 95, 105, 115, 125]
model = LinearRegression()
model.fit(X, y)
new_house = [[1700, 4, 3]]
prediction = model.predict(new_house)
print("House Price Prediction")
print("House Size : 1700 sq.ft")
print("Bedrooms   : 4")
print("House Age  : 3 years")
print("Predicted Price: ₹", round(prediction[0], 2), "Lakhs")
print("-----------------")

from sklearn.linear_model import LinearRegression
X = [[2, 70, 60],
    [3, 75, 65],
    [4, 80, 70],
    [5, 82, 75],
    [6, 85, 80],
    [7, 88, 85],
    [8, 90, 90],
    [9, 92, 94],
    [10, 95, 96],
    [11, 97, 98]]
y = [55, 60, 68, 74, 80, 86, 91, 95, 98, 100]
model = LinearRegression()
model.fit(X, y)
new_student = [[7, 90, 88]]
prediction = model.predict(new_student)
print("Student Marks Prediction")
print("Study Hours      : 7")
print("Attendance       : 90%")
print("Assignment Score : 88")
print("Predicted Final Marks:", round(prediction[0], 2))
print("------------------------")

from sklearn.linear_model import LinearRegression
X = [[1, 15, 60],
    [2, 15, 65],
    [3, 16, 70],
    [4, 16, 75],
    [5, 17, 80],
    [6, 17, 82],
    [7, 18, 85],
    [8, 18, 88],
    [9, 18, 90],
    [10, 19, 94]]
y = [3.5, 4.2, 5.1, 6.0, 7.2, 8.0, 9.1, 10.0, 11.2, 12.5]
model = LinearRegression()
model.fit(X, y)
new_employee = [[7, 18, 86]]
prediction = model.predict(new_employee)
print("Employee Salary Prediction")
print("Experience       : 7 years")
print("Education Level  : 18 years")
print("Skill Score      : 86")
print("Predicted Salary : ₹", round(prediction[0], 2), "Lakhs")
print("--------------------------")

from sklearn.linear_model import LinearRegression
X = [[1.0, 900, 60],
    [1.2, 950, 70],
    [1.4, 1050, 80],
    [1.6, 1150, 90],
    [1.8, 1250, 100],
    [2.0, 1350, 110],
    [2.2, 1450, 120],
    [2.4, 1550, 130],
    [2.6, 1650, 140],
    [2.8, 1750, 150]]
y = [24, 22, 20, 18, 16, 15, 13, 12, 11, 10]
model = LinearRegression()
model.fit(X, y)
new_car = [[1.5, 1100, 85]]
prediction = model.predict(new_car)
print("Car Fuel Efficiency Prediction")
print("Engine Capacity : 1.5 L")
print("Vehicle Weight  : 1100 kg")
print("Speed           : 85 km/h")
print("Predicted Mileage:", round(prediction[0], 2), "km/L")
print("------------------------------")

from sklearn.linear_model import LinearRegression
X = [[10, 2, 120],
    [15, 3, 150],
    [20, 4, 180],
    [25, 5, 210],
    [30, 6, 240],
    [35, 7, 260],
    [40, 8, 280],
    [45, 9, 300],
    [50, 10, 320],
    [55, 11, 340]]
y = [3.5, 4.5, 5.8, 7.0, 8.2, 9.5, 10.8, 12.0, 13.5, 15.0]
model = LinearRegression()
model.fit(X, y)
new_sales_data = [[38, 7, 270]]
prediction = model.predict(new_sales_data)
print("Sales Prediction")
print("Advertising Cost : ₹38 Thousand")
print("Salespersons     : 7")
print("Store Visits     : 270")
print("Predicted Monthly Sales: ₹", round(prediction[0], 2), "Lakhs")

