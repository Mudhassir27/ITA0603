from sklearn.tree import DecisionTreeClassifier
X=[[500],[600],[700],[800]]
y=['Poor','Fair','Good','Excellent']
m=DecisionTreeClassifier()
m.fit(X,y)
print(m.predict([[750]]))
print("-----------------")

from sklearn.tree import DecisionTreeClassifier
X=[[12,10,1,1],
[10,8,2,1],
[8,6,2,1],
[6,4,3,0],
[4,2,4,0],
[15,12,1,1],
[7,5,3,1],
[5,3,4,0],
[11,9,2,1],
[3,1,5,0]]
y=[
"Excellent",
"Good",
"Good",
"Average",
"Poor",
"Excellent",
"Average",
"Poor",
"Good",
"Poor"]
model=DecisionTreeClassifier(random_state=0)
model.fit(X,y)
new_sample=[[9,7,2,1]]
prediction=model.predict(new_sample)
print("Predicted Credit Score:",prediction[0])
print("-------------------")

from sklearn.tree import DecisionTreeClassifier
X=[[90000,45,5,0],
[75000,40,6,0],
[60000,35,8,0],
[45000,30,10,1],
[30000,25,12,1],
[95000,48,4,0],
[65000,37,7,0],
[40000,28,9,1],
[28000,24,13,1],
[85000,43,5,0]]
y=[
"Excellent",
"Good",
"Good",
"Average",
"Poor",
"Excellent",
"Good",
"Average",
"Poor",
"Excellent"
]
model=DecisionTreeClassifier(random_state=0)
model.fit(X,y)
new_sample=[[70000,38,6,0]]
prediction=model.predict(new_sample)
print("Predicted Credit Score:",prediction[0])
print("--------------------")

from sklearn.tree import DecisionTreeClassifier
X=[[8,40000,0],
[7,35000,2],
[6,30000,4],
[5,25000,10],
[4,20000,20],
[9,45000,0],
[6,28000,5],
[5,23000,12],
[3,18000,25],
[8,42000,1]]
y=[
"Excellent",
"Good",
"Good",
"Average",
"Poor",
"Excellent",
"Good",
"Average",
"Poor",
"Excellent"
]
model=DecisionTreeClassifier(random_state=0)
model.fit(X,y)
new_sample=[[7,36000,3]]
prediction=model.predict(new_sample)
print("Predicted Credit Score:",prediction[0])
print("----------------------")

from sklearn.tree import DecisionTreeClassifier

X=[[10,30000,1],
[8,35000,2],
[6,40000,3],
[4,45000,5],
[2,50000,7],
[11,28000,1],
[7,36000,2],
[5,43000,4],
[3,48000,6],
[9,32000,1]]
y=[
"Excellent",
"Good",
"Good",
"Average",
"Poor",
"Excellent",
"Good",
"Average",
"Poor",
"Excellent"
]
model=DecisionTreeClassifier(random_state=0)
model.fit(X,y)
new_sample=[[7,34000,2]]
prediction=model.predict(new_sample)
print("Predicted Credit Score:",prediction[0])
print("----------------------")

from sklearn.tree import DecisionTreeClassifier
X=[[15,850,1,1],
[12,800,2,1],
[10,760,2,1],
[8,700,3,0],
[6,620,4,0],
[16,880,1,1],
[11,780,2,1],
[7,680,3,0],
[5,600,5,0],
[14,830,1,1]]
y=[
"Excellent",
"Good",
"Good",
"Average",
"Poor",
"Excellent",
"Good",
"Average",
"Poor",
"Excellent"
]
model=DecisionTreeClassifier(random_state=0)
model.fit(X,y)
new_sample=[[11,790,2,1]]
prediction=model.predict(new_sample)
print("Predicted Credit Score:",prediction[0])
