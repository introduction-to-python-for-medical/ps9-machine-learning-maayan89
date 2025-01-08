import pandas as pd
df = pd.read_csv('parkinson.csv')
df = df.dropna()
feats = ['PPE','DFA']
tg = 'Status'
x = df[feats]
y = df[tg]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
xscale = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xscale, y, text_size=0.3, random_state=42)

from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
ypred = model.predict(x_test)
accuracy = accuracy_score(y_test, ypred)
print(accuracy)
