from sklearn.neural_network import MLPClassifier
X=[[0,0],[0,1],[1,0],[1,1]]
y=[0,1,1,0]
ann=MLPClassifier(hidden_layer_sizes=(2,),max_iter=1000)
ann.fit(X,y)
print("Prediction:",ann.predict([[1,0]]))
print("-----------")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
X = [
    [9, 2, 1, 2],
    [8, 1, 1, 1],
    [7, 1, 1, 0],
    [6, 0, 0, 0],
    [5, -1, 0, -1],
    [9, 2, 1, 1],
    [8, 1, 0, 1],
    [6, 0, 1, 0],
    [7, 2, 1, 1],
    [5, -1, 0, 0]
]
y = [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ann = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
ann.fit(X_scaled, y)
new_sample = [[8, 2, 1, 1]]
new_sample_scaled = scaler.transform(new_sample)
prediction = ann.predict(new_sample_scaled)
if prediction[0] == 1:
    print("Predicted Result: Placed")
else:
    print("Predicted Result: Not Placed")
print("------------")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
X = [
    [2, 2, 1, 1],
    [2, 2, 1, 0],
    [1, 2, 1, 1],
    [1, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [2, 1, 1, 1],
    [1, 2, 0, 0],
    [2, 2, 1, 1],
    [0, 0, 0, 1]
]
y = [1, 1, 1, 1, 0, 0, 1, 0, 1, 0]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ann = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
ann.fit(X_scaled, y)
new_sample = [[1, 2, 1, 1]]
new_sample_scaled = scaler.transform(new_sample)
prediction = ann.predict(new_sample_scaled)
if prediction[0] == 1:
    print("Predicted Result: Loan Approved")
else:
    print("Predicted Result: Loan Not Approved")
print("--------------")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
X = [
    [1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0]
]
y = [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ann = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
ann.fit(X_scaled, y)
new_sample = [[1, 1, 0, 1, 1]]
new_sample_scaled = scaler.transform(new_sample)
prediction = ann.predict(new_sample_scaled)
if prediction[0] == 1:
    print("Predicted Result: Positive")
else:
    print("Predicted Result: Negative")
print("-----------")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
X = [
    [2, 3, 1, 1],
    [2, 2, 1, 1],
    [1, 2, 1, 1],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [2, 3, 1, 0],
    [1, 2, 0, 1],
    [0, 1, 0, 0],
    [2, 2, 1, 1],
    [0, 0, 0, 1]
]
y = [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ann = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
ann.fit(X_scaled, y)
new_sample = [[1, 2, 1, 1]]
new_sample_scaled = scaler.transform(new_sample)
prediction = ann.predict(new_sample_scaled)
if prediction[0] == 1:
    print("Predicted Result: Promoted")
else:
    print("Predicted Result: Not Promoted")
print("----------")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
X = [
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 1, 0]
]
y = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ann = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
ann.fit(X_scaled, y)
new_sample = [[1, 1, 1, 0, 1]]
new_sample_scaled = scaler.transform(new_sample)
prediction = ann.predict(new_sample_scaled)
if prediction[0] == 1:
    print("Predicted Result: Spam")
else:
    print("Predicted Result: Not Spam")
