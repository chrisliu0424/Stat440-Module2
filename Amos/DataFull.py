# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:28:51 2020

@author: wlian
"""

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


class LinearBlock(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.fc = nn.Linear(in_c, out_c)
        self.norm = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(out_c, out_c)
        self.norm1 = nn.BatchNorm1d(out_c)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(out_c, out_c)
        self.norm2 = nn.BatchNorm1d(out_c)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(0.3)
        
        self.id = nn.Linear(in_c, out_c)
        
        self.relu3 = nn.ReLU()
        self.norm3 = nn.Dropout(0.3)
        
    def forward(self,x):
        y = self.fc(x)
        y = self.relu(y)
        # y = self.norm(y)
        y = self.do(y)
        
        y = self.fc1(y)
        y = self.relu1(y)
        # y = self.norm1(y)
        y = self.do1(y)
        
        y = self.fc2(y)
        y = self.relu2(y)
        # y = self.norm2(y)
        y = self.do2(y)
        
        y = y + self.id(x)
        # y = self.relu3(y)
        return y

class Net(nn.Module):
  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = LinearBlock(n_features,128).cuda()
    self.fc2 = LinearBlock(128,128).cuda()
    self.fc3 = LinearBlock(128,256).cuda()
    self.fc4 = LinearBlock(256,256).cuda()
    self.fc5 = LinearBlock(256,256).cuda()
    self.fc6 = LinearBlock(256,256).cuda()
    self.fc7 = LinearBlock(256,256).cuda()
    self.fc8 = LinearBlock(256,256).cuda()
    self.fc9 = LinearBlock(256,256).cuda()
    self.fc10 = LinearBlock(256,128).cuda()
    self.fc11 = LinearBlock(128,128).cuda()
    self.fc12 = LinearBlock(128,128).cuda()
    self.fc13 = LinearBlock(128,128).cuda()
    self.fc14 = LinearBlock(128,128).cuda()
    self.fc15 = LinearBlock(128,128).cuda()
    self.fc16 = LinearBlock(128,128).cuda()
    self.fc17 = LinearBlock(128,128).cuda()
    self.fc18 = LinearBlock(128,128).cuda()
    self.fc19 = nn.Linear(128, 14)
    
  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.fc4(x)
    x = self.fc5(x)
    x = self.fc6(x)
    x = self.fc7(x)
    x = self.fc8(x)
    x = self.fc9(x)
    x = self.fc10(x)
    x = self.fc11(x)
    x = self.fc12(x)
    x = self.fc13(x)
    x = self.fc14(x)
    x = self.fc15(x)
    x = self.fc16(x)
    x = self.fc17(x)
    x = self.fc18(x)
    return self.fc19(x)

result = pd.DataFrame()
x_train, x_valid, y_train, y_valid = train_test_split( x_train, y_train , test_size=0.20)

test = torch.Tensor(np.array(x_test))
x_train_tensor = torch.Tensor(np.array(x_train))
x_valid_tensor = torch.Tensor(np.array(x_valid))

os.chdir("../Amos")
criterion = nn.MSELoss()
model = Net(x_train.shape[1]).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005,weight_decay=0.0001)


model.train()
model.cuda()
y_tensor = y_train
y_tensor = torch.Tensor(np.array(y_tensor))

epoch = 200

train = TensorDataset(x_train_tensor,y_tensor)
train_loader = DataLoader(train, batch_size = 512, shuffle = True)
for epoch in range(epoch):
    running_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_batch.cuda())
        # Compute Loss
        loss = criterion(y_pred, y_batch.cuda())
       
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    running_loss/=len(train_loader)
    print('Epoch {}: train loss: {}'.format(epoch, running_loss))
    
    if(epoch%5 == 0):
        model.to('cpu')
        model.eval()
        y_valid_pred =  model(x_valid_tensor)
        y_valid_pred = y_valid_pred.detach().numpy()
        y_valid_pred = pd.DataFrame(y_valid_pred,columns=([y_valid.columns]))
        mse = mean_squared_error(y_valid,y_valid_pred)
        print('Result : MSE: {}'.format(mse))
        model.cuda()
        model.train()

del y_tensor

model.to('cpu')
PATH = "./model" + ".pth" 
torch.save(model.state_dict(), PATH)

model.eval()
y_predid = model(test)
y_predid = y_predid.detach().numpy()
y_predid = pd.DataFrame(y_predid,columns=([y_train.columns]))
result = pd.concat([result.reset_index(drop=True), y_predid], axis=1)

y_valid_pred =  model(x_valid_tensor)
y_valid_pred = y_valid_pred.detach().numpy()
y_valid_pred = pd.DataFrame(y_valid_pred,columns=([y_valid.columns]))
mse = mean_squared_error(y_valid,y_valid_pred)
print('Result : MSE: {}'.format(mse))

result.to_csv("./results_full.csv",header=True,index=False)
del x_train_tensor
del test
del model



