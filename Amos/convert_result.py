# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:33:02 2020

@author: wlian
"""
import os
import pandas as pd
os.chdir("../original_data")


x_test = pd.read_csv("Xtest.txt",delimiter=' ',dtype={"B15": str})
y_test = pd.read_csv("Ytest.txt",delimiter=',')


os.chdir("../Amos")
y_test[["Id","d"]] = y_test.Id.str.split(":",expand=True)

y_test = y_test.astype({'Id': int,"d":str})

results_final = pd.read_csv("results_full.csv")
results_final = pd.concat([results_final.reset_index(drop=True), x_test.Id], axis=1)
results_final = results_final.astype({'Id': int})

for i in range(results_final.shape[0]):
    if y_test.Id[i] == results_final.Id[i]:
        y_test.Value[i] = results_final[y_test.d[i]][i]


y_test = y_test.drop(['d'], axis=1)
y_test.to_csv("./results_kaggle_1.csv",header=True,index=False)
