from sklearn.linear_model import LinearRegression
X=[[500],[1000],[1500],[2000]]
y=[20,40,60,80]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[2500]]))
print("---------------------")


from sklearn.linear_model import LinearRegression
X=[
[800,2,15],
[1000,2,12],
[1200,3,10],
[1400,3,8],
[1600,4,6],
[1800,4,5],
[2000,5,4],
[2200,5,3],
[2400,6,2],
[2600,6,1]
]
y=[35,42,52,63,75,88,102,118,135,150]
model=LinearRegression()
model.fit(X,y)
new_sample=[[1700,4,5]]
prediction=model.predict(new_sample)
print("Predicted House Price:",round(prediction[0],2),"Lakhs")
print("---------------------")


from sklearn.linear_model import LinearRegression
X=[
[900,20,1],
[1100,18,1],
[1300,16,2],
[1500,14,2],
[1700,12,2],
[1900,10,3],
[2100,8,3],
[2300,6,4],
[2500,5,4],
[2700,3,5]
]
y=[40,48,60,74,88,104,122,142,165,190]
model=LinearRegression()
model.fit(X,y)
new_sample=[[1800,11,3]]
prediction=model.predict(new_sample)
print("Predicted House Price:",round(prediction[0],2),"Lakhs")
print("-------------------")

from sklearn.linear_model import LinearRegression
X=[
[1000,1,100],
[1200,1,120],
[1400,2,150],
[1600,2,180],
[1800,2,200],
[2000,3,240],
[2200,3,280],
[2400,3,320],
[2600,4,350],
[2800,4,400]
]
y=[45,55,68,82,96,112,130,150,172,196]
model=LinearRegression()
model.fit(X,y)
new_sample=[[2100,3,250]]
prediction=model.predict(new_sample)
print("Predicted House Price:",round(prediction[0],2),"Lakhs")
print("-------------------")

from sklearn.linear_model import LinearRegression
X=[
[700,1,15],
[850,2,12],
[1000,3,10],
[1150,4,8],
[1300,5,7],
[1450,6,5],
[1600,7,4],
[1750,8,3],
[1900,9,2],
[2050,10,1]
]
y=[30,38,48,60,72,86,100,116,134,154]
model=LinearRegression()
model.fit(X,y)
new_sample=[[1500,6,4]]
prediction=model.predict(new_sample)
print("Predicted Apartment Price:",round(prediction[0],2),"Lakhs")
print("----------------------")


from sklearn.linear_model import LinearRegression
X=[
[1200,900,2],
[1400,1100,2],
[1600,1300,3],
[1800,1500,3],
[2000,1700,4],
[2200,1900,4],
[2400,2100,5],
[2600,2300,5],
[2800,2500,6],
[3000,2700,6]
]
y=[48,60,74,90,108,128,150,174,200,228]
model=LinearRegression()
model.fit(X,y)
new_sample=[[2300,2000,4]]
prediction=model.predict(new_sample)
print("Predicted House Price:",round(prediction[0],2),"Lakhs")
