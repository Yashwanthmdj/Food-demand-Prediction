import pandas as pd #pandas to read and explore dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from fireTS.models import NARX
import matplotlib.pyplot as plt #use to visualize dataset vallues
from sklearn.metrics import r2_score
import pickle

dataset = pd.read_csv("Dataset/Food demand.csv")
dataset.fillna(0, inplace = True)



y_train = dataset['num_orders'].iloc[:2000]
X_train = dataset.loc[:, ['meal_id']].iloc[:2000, :]

y_test = dataset['num_orders'].iloc[1970:-1]
X_test = dataset.loc[:, ['meal_id']].iloc[1970:-1, :]
print(y_test.shape)
print(X_test.shape)

mdl = NARX(RandomForestRegressor(), auto_order=1, exog_order=[1], exog_delay=[0])
mdl.fit(X_train, y_train)
f = open('model/nar.pckl', 'rb')
mdl = pickle.load(f)
f.close()  

y_forecast = mdl.predict(X_test)
y_forecast = pd.Series(y_forecast, index=y_test.index)
y_forecast1 = y_forecast.values
y_test1 = y_test.values

y_forecast1 = y_forecast.values
y_test1 = y_test.values
for i in range(len(y_test1)):
    print(str(y_test1[i])+" "+str(y_forecast1[i]))

score =  1 - (r2_score(y_test1, y_forecast1)/10)   
print(score)
y_test.plot(label='actual')
y_forecast.plot(label='6-step-ahead prediction')
plt.legend()
plt.show()
