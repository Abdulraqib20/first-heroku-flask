import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')

X = df.drop(['target'], axis=1)
y = df['target']

lr_model = LogisticRegression()
lr_model.fit(X, y)

joblib.dump(lr_model, 'lr_model.joblib')
load_model = joblib.load('lr_model.joblib')
print(load_model.predict([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]))