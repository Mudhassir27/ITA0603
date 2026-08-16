from sklearn.linear_model import LinearRegression
X=[[1],[2],[3],[4]]
y=[100000,200000,300000,400000]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[5]]))
print("-----------------")

from sklearn.linear_model import LinearRegression
X=[
[1,12000,1200],
[2,18000,1200],
[3,25000,1300],
[4,35000,1400],
[5,45000,1500],
[6,55000,1500],
[7,65000,1600],
[8,75000,1600],
[9,85000,1800],
[10,95000,1800]
]
y=[8.5,8.0,7.4,6.8,6.1,5.5,5.0,4.5,4.0,3.5]
model=LinearRegression()
model.fit(X,y)
new_sample=[[4,38000,1400]]
prediction=model.predict(new_sample)
print("Predicted Car Price:",round(prediction[0],2),"Lakhs")
print("------------------")

from sklearn.linear_model import LinearRegression
X=[
[2024,10000,22],
[2023,18000,21],
[2022,25000,20],
[2021,35000,19],
[2020,45000,18],
[2019,55000,17],
[2018,65000,16],
[2017,75000,15],
[2016,85000,14],
[2015,95000,13]
]
y=[10.5,9.8,9.0,8.2,7.5,6.8,6.0,5.4,4.8,4.2]
model=LinearRegression()
model.fit(X,y)
new_sample=[[2021,32000,19]]
prediction=model.predict(new_sample)
print("Predicted Car Price:",round(prediction[0],2),"Lakhs")
print("---------------------")

from sklearn.linear_model import LinearRegression
X=[
[1000,70,10000],
[1200,82,15000],
[1400,95,22000],
[1500,105,30000],
[1600,115,38000],
[1800,130,45000],
[2000,145,52000],
[2200,160,60000],
[2400,175,68000],
[2600,190,76000]
]
y=[6.5,7.4,8.8,9.8,10.5,11.8,13.2,14.5,15.8,17.2]
model=LinearRegression()
model.fit(X,y)
new_sample=[[1700,120,42000]]
prediction=model.predict(new_sample)
print("Predicted Car Price:",round(prediction[0],2),"Lakhs")
print("---------------------")

from sklearn.linear_model import LinearRegression
X=[
[1500,1,12000],
[1800,2,20000],
[2000,3,28000],
[2200,4,36000],
[2400,5,44000],
[2600,6,52000],
[2800,7,60000],
[3000,8,68000],
[3200,9,76000],
[3400,10,84000]
]
y=[15.5,14.8,13.9,13.0,12.0,11.0,10.2,9.4,8.7,8.0]
model=LinearRegression()
model.fit(X,y)
new_sample=[[2100,4,34000]]
prediction=model.predict(new_sample)
print("Predicted SUV Price:",round(prediction[0],2),"Lakhs")
print("-----------------------")

from sklearn.linear_model import LinearRegression
X=[
[30,250,8],
[35,300,8],
[40,350,7],
[45,400,7],
[50,450,6],
[55,500,6],
[60,550,5],
[65,600,5],
[70,650,4],
[75,700,4]
]
y=[10.5,12.0,14.0,16.5,18.5,21.0,24.0,27.5,31.0,35.0]
model=LinearRegression()
model.fit(X,y)
new_sample=[[52,470,6]]
prediction=model.predict(new_sample)
print("Predicted Electric Car Price:",round(prediction[0],2),"Lakhs")
