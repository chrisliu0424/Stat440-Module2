# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:08:12 2020

@author: wlian
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

os.chdir("../original_data")

x_train = pd.read_csv("Xtrain.txt",delimiter=' ',dtype={"B15": str})
y_train = pd.read_csv("Ytrain.txt",delimiter=' ')

x_test = pd.read_csv("Xtest.txt",delimiter=' ',dtype={"B15": str})
y_test = pd.read_csv("Ytest.txt",delimiter=' ')


x_train = x_train.replace('?', np.nan)
x_test = x_test.replace('?', np.nan)

x_train = x_train.astype({'B15': 'float64'})
x_test = x_test.astype({'B15': 'float64'})

x_train = x_train.fillna(x_train.median())
x_test = x_test.fillna(x_test.median())

x_train = x_train[(np.abs(stats.zscore(x_train)) < 4).all(axis=1)]

df_train = x_train.merge(y_train, left_on='Id', right_on='Id')

df_train = df_train.fillna(df_train.median())


#Plot 
'''

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(0, 250, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},xticklabels=True, yticklabels=True)
'''
#


x_train = df_train.loc[:, :'F08']
y_train = df_train.loc[:,"Z01" :'Z14']
x_test = x_test.loc[:, :'F08']


corr = x_train.corr()
toppair = corr.unstack().abs().sort_values().drop_duplicates()

x_train.drop(['C01', 'D04', "D09", "Id"], axis=1, inplace=True)
x_test.drop(['C01', 'D04', "D09", "Id"], axis=1, inplace=True)

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")

class Net(nn.Module):
  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 100)
    self.fc2 = nn.Linear(100, 200)
    self.fc3 = nn.Linear(200, 100)
    self.fc7 = nn.Linear(100, 1)
    
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return self.fc7(x)



result = pd.DataFrame()
x_train, x_valid, y_train, y_valid = train_test_split( x_train, y_train , test_size=0.20, random_state=42)

test = torch.Tensor(np.array(x_test))
x_train_tensor = torch.Tensor(np.array(x_train))
x_valid_tensor = torch.Tensor(np.array(x_valid))

os.chdir("../Amos")
criterion = nn.MSELoss()
model = Net(x_train.shape[1]).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001,momentum=0.9,weight_decay=1e-05)

column_i = 0
model.train()
model.cuda()
path = "./" + str(y_train.columns[column_i]) + ".pth" 
y_tensor = y_train[y_train.columns[column_i]]
y_tensor = torch.Tensor(np.array(y_tensor))
y_valid_loop =y_valid[y_valid.columns[column_i]]

epoch = 25

train = TensorDataset(x_train_tensor,y_tensor)
train_loader = DataLoader(train, batch_size = 256, shuffle = True)
for epoch in range(epoch):
    running_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_batch.cuda())
        # Compute Loss
        loss = criterion(y_pred, y_batch.cuda().unsqueeze(1))
       
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    running_loss/=len(train_loader)
    print('Epoch {}: train loss: {}'.format(epoch, running_loss))

del y_tensor

model.to('cpu')
PATH = str(y_train.columns[column_i])+".pth"
torch.save(model.state_dict(), PATH)

model.eval()
y_predid = model(test)
y_predid = y_predid.detach().numpy()
y_predid = pd.DataFrame(y_predid,columns=([str(y_train.columns[column_i])]))
result = pd.concat([result.reset_index(drop=True), y_predid], axis=1)

y_valid_pred =  model(x_valid_tensor)
y_valid_pred = y_valid_pred.detach().numpy()
col_name = "Z"+str(column_i+1)
y_valid_pred = pd.DataFrame(y_valid_pred,columns=([col_name]))
mse = mean_squared_error(y_valid_loop,y_valid_pred)
print('Column {}: MSE: {}'.format(str(y_valid_pred.columns[0]), mse))

result.to_csv("./results.csv",header=True,index=False)
del x_train_tensor
del test
del model



