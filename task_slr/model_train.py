import pickle
from statistics import linear_regression, LinearRegression

import pandas as pd
import numpy as np
from numpy.random.mtrand import random_sample
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
df=pd.read_csv("data.csv")
x=df[["Height"]]
y=df["Weight"]
xtrain,xtest,ytrain,ytest=train_test_split(
    x,y,
    test_size=0.25,random_state=42
)
model=LinearRegression()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
r2=r2_score(ytest,pred)
print(r2)
with open("task_slr.pkl","wb" )as file:
    pickle.dump(model,file)