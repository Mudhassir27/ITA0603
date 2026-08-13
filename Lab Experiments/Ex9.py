from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X=[[1],[2],[3],[4],[5]]
y=[1,4,9,16,25]
print("Linear:", LinearRegression().fit(X,y).predict(X))
X2=PolynomialFeatures(2).fit_transform(X)
print("Poly:", LinearRegression().fit(X2,y).predict(X2))
print("-----------------")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X = [[800], [1000], [1200], [1400], [1600],
     [1800], [2000], [2200], [2400], [2600]]
y = [32, 40, 52, 68, 88, 112, 140, 172, 208, 248]
linear_model = LinearRegression()
linear_model.fit(X, y)
new_sample = [[1700]]
linear_prediction = linear_model.predict(new_sample)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
new_sample_poly = poly.transform(new_sample)
poly_prediction = poly_model.predict(new_sample_poly)
print("House Price Prediction")
print("House Size:", 1700, "sq.ft")
print("Linear Regression Prediction: ₹", round(linear_prediction[0], 2), "Lakhs")
print("Polynomial Regression Prediction: ₹", round(poly_prediction[0], 2), "Lakhs")
print("----------------------")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X = [[1], [2], [3], [4], [5],
     [6], [7], [8], [9], [10]]
y = [18, 30, 43, 57, 68, 78, 86, 92, 96, 98]
linear_model = LinearRegression()
linear_model.fit(X, y)
new_sample = [[7.5]]
linear_prediction = linear_model.predict(new_sample)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
new_sample_poly = poly.transform(new_sample)
poly_prediction = poly_model.predict(new_sample_poly)
print("Student Marks Prediction")
print("Study Hours:", 7.5)
print("Linear Regression Prediction:", round(linear_prediction[0], 2), "%")
print("Polynomial Regression Prediction:", round(poly_prediction[0], 2), "%")
print("------------------------")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X = [[40], [50], [60], [70], [80],
     [90], [100], [110], [120], [130]]
y = [4.5, 4.8, 5.2, 5.9, 6.8, 8.0, 9.5, 11.2, 13.4, 16.0]
linear_model = LinearRegression()
linear_model.fit(X, y)
new_sample = [[95]]
linear_prediction = linear_model.predict(new_sample)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
new_sample_poly = poly.transform(new_sample)
poly_prediction = poly_model.predict(new_sample_poly)
print("Car Fuel Consumption Prediction")
print("Speed:", 95, "km/h")
print("Linear Regression Prediction:", round(linear_prediction[0], 2), "L/100 km")
print("Polynomial Regression Prediction:", round(poly_prediction[0], 2), "L/100 km")
print("--------------------------------")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X = [[5], [10], [15], [20], [25],
     [30], [35], [40], [45], [50]]
y = [1.2, 2.5, 4.1, 6.2, 8.8, 11.9, 15.5, 19.6, 24.2, 29.3]
linear_model = LinearRegression()
linear_model.fit(X, y)
new_sample = [[32]]
linear_prediction = linear_model.predict(new_sample)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
new_sample_poly = poly.transform(new_sample)
poly_prediction = poly_model.predict(new_sample_poly)
print("Advertising Cost vs Sales")
print("Advertising Cost: ₹", 32, "Thousand")
print("Linear Regression Prediction: ₹", round(linear_prediction[0], 2), "Lakhs")
print("Polynomial Regression Prediction: ₹", round(poly_prediction[0], 2), "Lakhs")
print("-------------------------")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X = [[1], [2], [3], [4], [5],
     [6], [7], [8], [9], [10]]
y = [3.0, 3.8, 4.9, 6.3, 8.0, 10.2, 12.9, 16.1, 19.8, 24.0]
linear_model = LinearRegression()
linear_model.fit(X, y)
new_sample = [[7.5]]
linear_prediction = linear_model.predict(new_sample)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
new_sample_poly = poly.transform(new_sample)
poly_prediction = poly_model.predict(new_sample_poly)
print("Salary Prediction")
print("Experience:", 7.5, "years")
print("Linear Regression Prediction: ₹", round(linear_prediction[0], 2), "Lakhs")
print("Polynomial Regression Prediction: ₹", round(poly_prediction[0], 2), "Lakhs")
