from sklearn.neural_network import MLPClassifier
X=[[0,0],[0,1],[1,0],[1,1]]
y=[0,1,1,0]
ann=MLPClassifier(hidden_layer_sizes=(2,),max_iter=1000)
ann.fit(X,y)
print("Prediction:",ann.predict([[1,0]]))
print("-----------")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
data = {
    "CGPA":[9,8,7,6,5,9,8,6,7,5],
    "Communication":["Excellent","Good","Good","Average","Poor","Excellent","Good","Average","Excellent","Poor"],
    "Internship":["Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No"],
    "Programming":["Excellent","Good","Average","Average","Poor","Good","Good","Average","Good","Average"],
    "Placement":["Placed","Placed","Placed","Not Placed","Not Placed","Placed","Placed","Not Placed","Placed","Not Placed"]
}
df = pd.DataFrame(data)
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
X = df.drop("Placement", axis=1)
y = df["Placement"]
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
model.fit(X, y)
new_sample = pd.DataFrame({
    "CGPA":[8],
    "Communication":["Excellent"],
    "Internship":["Yes"],
    "Programming":["Good"]
})
for col in new_sample.columns:
    if col != "CGPA":
        new_sample[col] = encoders[col].transform(new_sample[col])
prediction = model.predict(new_sample)
print("Predicted Placement:",
      encoders["Placement"].inverse_transform(prediction)[0])
print("-----------")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
data = {
    "Income":["High","High","Medium","Medium","Low","Low","High","Medium","High","Low"],
    "Credit Score":["Good","Good","Good","Average","Poor","Average","Average","Good","Good","Poor"],
    "Employment":["Permanent","Permanent","Permanent","Permanent","Temporary","Temporary","Permanent","Temporary","Permanent","Temporary"],
    "Property":["Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes"],
    "Loan Approved":["Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No"]
}
df = pd.DataFrame(data)
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
X = df.drop("Loan Approved", axis=1)
y = df["Loan Approved"]
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
model.fit(X, y)
new_sample = pd.DataFrame({
    "Income":["Medium"],
    "Credit Score":["Good"],
    "Employment":["Permanent"],
    "Property":["Yes"]
})
for col in new_sample.columns:
    new_sample[col] = encoders[col].transform(new_sample[col])
prediction = model.predict(new_sample)
print("Loan Approval:",
      encoders["Loan Approved"].inverse_transform(prediction)[0])
print("-----------")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
data = {
    "Fever":["Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No"],
    "Cough":["Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No"],
    "Headache":["Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes"],
    "Body Pain":["Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No"],
    "Fatigue":["Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No"],
    "Disease":["Positive","Positive","Negative","Positive","Negative","Positive","Negative","Positive","Positive","Negative"]
}
df = pd.DataFrame(data)
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
X = df.drop("Disease", axis=1)
y = df["Disease"]
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
model.fit(X, y)
new_sample = pd.DataFrame({
    "Fever":["Yes"],
    "Cough":["Yes"],
    "Headache":["No"],
    "Body Pain":["Yes"],
    "Fatigue":["Yes"]
})
for col in new_sample.columns:
    new_sample[col] = encoders[col].transform(new_sample[col])
prediction = model.predict(new_sample)
print("Disease Prediction:",
      encoders["Disease"].inverse_transform(prediction)[0])
print("-----------")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
data = {
    "Experience":["High","High","Medium","Medium","Low","High","Medium","Low","High","Low"],
    "Performance":["Excellent","Good","Good","Average","Poor","Excellent","Good","Average","Good","Poor"],
    "Leadership":["Yes","Yes","Yes","No","No","Yes","No","No","Yes","No"],
    "Training":["Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes"],
    "Promotion":["Promoted","Promoted","Promoted","Not Promoted","Not Promoted","Promoted","Promoted","Not Promoted","Promoted","Not Promoted"]
}
df = pd.DataFrame(data)
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
X = df.drop("Promotion", axis=1)
y = df["Promotion"]
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
model.fit(X, y)
new_sample = pd.DataFrame({
    "Experience":["Medium"],
    "Performance":["Good"],
    "Leadership":["Yes"],
    "Training":["Yes"]
})
for col in new_sample.columns:
    new_sample[col] = encoders[col].transform(new_sample[col])
prediction = model.predict(new_sample)
print("Promotion Prediction:",
      encoders["Promotion"].inverse_transform(prediction)[0])
print("-----------")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
data = {
    "Contains Link":["Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No"],
    "Offer Words":["Yes","Yes","No","No","Yes","Yes","No","No","Yes","No"],
    "Unknown Sender":["Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No"],
    "Attachment":["No","Yes","No","No","Yes","No","Yes","No","No","Yes"],
    "Many Recipients":["Yes","Yes","No","Yes","No","No","Yes","No","Yes","No"],
    "Spam":["Spam","Spam","Not Spam","Spam","Not Spam","Spam","Spam","Not Spam","Spam","Not Spam"]
}
df = pd.DataFrame(data)
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
X = df.drop("Spam", axis=1)
y = df["Spam"]
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
model.fit(X, y)
new_sample = pd.DataFrame({
    "Contains Link":["Yes"],
    "Offer Words":["Yes"],
    "Unknown Sender":["Yes"],
    "Attachment":["No"],
    "Many Recipients":["Yes"]
})
for col in new_sample.columns:
    new_sample[col] = encoders[col].transform(new_sample[col])
prediction = model.predict(new_sample)
print("Email Classification:",
      encoders["Spam"].inverse_transform(prediction)[0])
print("-----------")
