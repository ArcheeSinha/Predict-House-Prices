import pandas as pd
import numpy as np
from sklearn import linear_model 

df=pd.read_csv("homeprices.csv")
df

import math
median_bedrooms=math.floor(df.bedrooms.median())
median_bedrooms

df.bedrooms=df.bedrooms.fillna(median_bedrooms)
(df.bedrooms)

reg = linear_model.LinearRegression()      #creating a linear regression object
reg.fit(df[['area','bedrooms','age']],df.price)    #fitting the model
print(reg.coef_)                               #getting the coefficients of the model
print(reg.intercept_)                         #getting the intercept of the model

prediction = reg.predict(pd.DataFrame([[3000,3, 40]], columns=['area', 'bedrooms', 'age']))   #predicting the price
print(prediction)

#print(137.25*3000+-26025*3+-6825*40 + 383724.99999999977)


