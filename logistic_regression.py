import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('train.csv')
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df.dropna(inplace=True)

df['Sex'] = LabelEncoder().fit_transform(df['Sex']) # Male=1, Female=0
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked']) # C=0, Q=1, S=2

X = df.drop('Survived', axis=1)
y = df['Survived']

scale = StandardScaler()
X_scaled = scale.fit_transform(X)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)


print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))


new_passenger = np.array([[3, 1, 33, 1, 0, 7.25, 0]])
new_passenger_scaled = scale.transform(new_passenger)

prediction = model.predict(new_passenger_scaled)
probability = model.predict_proba(new_passenger_scaled)

print('prediction:', 'will die' if prediction[0] == 0 else 'will survive')
print('probability of death:', f'{probability[0][0]:.2%}')
print('probability of survival:', f'{probability[0][1]:.2%}')