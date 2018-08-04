# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 01:45:24 2018

@author: KARIS
"""

import pandas as pd
features_test = pd.read_csv("dengue_features_test.csv")
features_train = pd.read_csv("dengue_features_train.csv")
labels_train = pd.read_csv("dengue_labels_train.csv")




for i in features_train :
    features_train[i] = features_train[i].fillna(features_train[i].mode()[0])
    
    
    
for i in features_test :
    features_test[i] = features_test[i].fillna(features_test[i].mode()[0])


features_test.drop(["city","year", "weekofyear" , "week_start_date" ],axis=1, inplace = True)

features_train.drop(["city","year", "weekofyear" , "week_start_date" ],axis=1, inplace = True)

labels_train.drop(["city","year", "weekofyear" ],axis=1, inplace = True)

'''
from sklearn.preprocessing import StandardScaler as SS
sc = SS()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)
'''
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(features_train , labels_train)


pred = pd.DataFrame((lm.predict(features_test)).astype(int))
test = pd.read_csv("dengue_features_test.csv")
test = test.iloc[: , :3]
df_new=pd.concat([test,pred],axis = 1)
df_new = df_new.rename(index=str, columns={0: "total_cases"})
df_new.to_csv("prediction.csv",index=False)
