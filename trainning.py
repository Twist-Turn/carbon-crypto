import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pickle
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering
from sklearn import linear_model

df=pd.read_csv("FuelConsumptionCo2.csv")

cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_xyz','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB_MPG','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=1068,
                                  random_state=0)
reggr=linear_model.LinearRegression()
k_regressor=KNeighborsRegressor(n_neighbors=1)
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['FUELCONSUMPTION_xyz']])
train_x1 = np.asanyarray(train[['ENGINESIZE']])
train_x2 = np.asanyarray(train[['CO2EMISSIONS']])
train_x3 = np.asanyarray(train[['CYLINDERS']])
train_y = np.asanyarray(train[['FUELCONSUMPTION_CITY']])
train_y1 = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
train_y2 = np.asanyarray(train[['FUELCONSUMPTION_COMB_MPG']])
train_y3 = np.asanyarray(train[['FUELCONSUMPTION_HWY']])
X=train_x1+train_x2+train_x3+train_x+train_y3+train_y+train_y2+train_y3+train_x+train_x1
Y=train_x2
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(train_x1,train_x2)

plt.scatter(train_x1, train_x2, c=labels)
plt.show()
reggr.fit(X,Y)
regressor.fit(X,Y)
k_regressor.fit(X,Y)
regr.fit(X,Y)
print(f"Accuracy score of K=Nearest-Neighbor is :{k_regressor.score(X,Y):.0%}")
print(f"Accuracy score of RandomForestClassifier is: {regressor.score(X,Y):.0%}")






test_x = np.asanyarray(test[['FUELCONSUMPTION_xyz']])
test_x1 = np.asanyarray(test[['ENGINESIZE']])
test_x2 = np.asanyarray(test[['CO2EMISSIONS']])
test_x3 = np.asanyarray(test[['CYLINDERS']])
test_y = np.asanyarray(test[['FUELCONSUMPTION_CITY']])
test_y1 = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y2 = np.asanyarray(test[['FUELCONSUMPTION_COMB_MPG']])
test_y3 = np.asanyarray(test[['FUELCONSUMPTION_HWY']])








#open a file where you want to store the data
file=open('model1.pkl','wb')
pickle.dump(reggr,file)
file.close()

