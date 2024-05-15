#!/usr/bin/env python

import pandas as pd
data = pd.read_csv('dataset.csv')

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Assuming no missing values or they are now handled, we move to scaling:
from sklearn.preprocessing import StandardScaler

# Features and Labels
X = data.drop('CLASS_LABEL', axis=1)
y = data['CLASS_LABEL']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))