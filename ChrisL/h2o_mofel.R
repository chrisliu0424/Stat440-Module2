library(h2o)
library(stringr)
rm(list = ls())
source("~/Documents/GitHub/Stat440-Module2/ChrisL/functions.R")

# Read in data

X_train = read.csv("~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/X_train_clean.csv")
X_test = read.csv("~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/X_test_clean.csv")
Y_train = read.csv("~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/Y_train_clean.csv")
Y_test = read.csv("~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/Y_test_clean.csv")

X_train = X_train[,-1]
Y_train = Y_train[,-1]
X_test = X_test[,-1]

train_data = cbind(Y_train,X_train)
h2o.init(nthreads = -1)

# Transfer 
train_hex = as.h2o(train_data)
X_test_hex = as.h2o(X_test)
prediction_matrix = matrix(NA,nrow = 50000,ncol = 14)
for(i in 1:14){
  print(i)
  gbm = h2o.gbm(x = 15:69,y = i, training_frame = train_hex, ntrees = 3000, learn_rate = 0.01)
  prediction = h2o.predict(gbm,newdata = X_test_hex)
  prediction_matrix[,i] = as.matrix(prediction)
}

prediction = get_matrix(prediction_matrix,Y_test)
write.csv(prediction,"kaggle2.csv",row.names = FALSE)
