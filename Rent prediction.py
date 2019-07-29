#Librairies import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
from statistics import mean


#Loading the dataset and eliminating missing data
house_data = pd.read_csv('C:/Users/HP/Desktop/mldata/house_data.csv').dropna()
#Aberrant data processing (outliers)
house_data = house_data[house_data['price']<10000]
#Dataset splitting to training/test sets
surface_train,surface_test,price_train,price_test = train_test_split(house_data['surface'],house_data['price'],train_size=0.8)
arrond_train,arrond_test = train_test_split(house_data['arrondissement'],train_size=0.8)
#converting pandas.series objects to dataframes
surface_train = surface_train.to_frame()
price_train = price_train.to_frame()
surface_test = surface_test.to_frame()
price_test = price_test.to_frame()

#creation and training of the first linear regression model with a single feature 
linreg1 = linear_model.LinearRegression()
linreg1.fit(surface_train,price_train)
score1 = linreg1.score(surface_test,price_test)*100
#display of the model score 
print("Les prédictions du modèle sont précises à %.2f" %score1, "%")

#First improvement proposal: linear regression model with two features (price and district) joined
x = pd.get_dummies(house_data['arrondissement'])
house_data2 = house_data.drop('arrondissement',axis=1)
house_data2 = pd.concat([house_data2,x],axis=1)
surface_arrond = house_data2.iloc[:,1:-1]
price = house_data2.iloc[:,:1]
#Dataset splitting to training/test sets
sa_train,sa_test,p_train,p_test = train_test_split(surface_arrond,price,train_size=0.8)
linreg2 = linear_model.LinearRegression()
linreg2.fit(sa_train,p_train)
score2 = linreg2.score(sa_test,p_test)*100
#showing the score of the first improvement proposal
print("Les prédictions du modèle amélioré 1 sont précises à %.2f" %score2, "%")

#second proposal for improvement of the model: linear regression model with two features with separation according to the districts

#step1: decomposition of the dataset into sub-datasets according to each district and their separation into data (surface and district) and target (price of the rent-price)
house_data_arrond1 = house_data2[house_data2[1]==1].drop('price',axis=1)
house_data_arrond1_price = house_data2[house_data2[1]==1].drop(['surface',1,2,3,4,10],axis=1)
house_data_arrond2 = house_data2[house_data2[2]==1].drop('price',axis=1)
house_data_arrond2_price = house_data2[house_data2[2]==1].drop(['surface',1,2,3,4,10],axis=1)
house_data_arrond3 = house_data2[house_data2[3]==1].drop('price',axis=1)
house_data_arrond3_price = house_data2[house_data2[3]==1].drop(['surface',1,2,3,4,10],axis=1)
house_data_arrond4 = house_data2[house_data2[4]==1].drop('price',axis=1)
house_data_arrond4_price = house_data2[house_data2[4]==1].drop(['surface',1,2,3,4,10],axis=1)
house_data_arrond10 = house_data2[house_data2[10]==1].drop('price',axis=1)
house_data_arrond10_price = house_data2[house_data2[10]==1].drop(['surface',1,2,3,4,10],axis=1)

all_data = [[house_data_arrond1,house_data_arrond1_price],[house_data_arrond2,house_data_arrond2_price],[house_data_arrond3,house_data_arrond3_price],[house_data_arrond4,house_data_arrond4_price],[house_data_arrond10,house_data_arrond10_price]]


score_list = []
for data in all_data:
    satr,sate,prtr,prte = train_test_split(data[0],data[1],train_size=0.8)
    model = linear_model.LinearRegression()
    model.fit(satr,prtr)
    model_score = model.score(sate,prte)
    score_list.append(model_score)
#step4: the score is returned as the average model score for each district

print("Les prédictions du modèle amélioré 2 sont précises à %.2f" %(mean(score_list)*100), "%")

#we run the program several times it turns out that often the second improved model displays higher scores than the first improved model
    
