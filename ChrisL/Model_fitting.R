rm(list = ls())
library(glmnet)
options(na.action='na.pass')

# Read in helper functions
source("~/Documents/GitHub/Stat440-Module2/ChrisL/functions.R")

# Read in data

X_train = read.csv("~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/X_train_clean.csv")
X_test = read.csv("~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/X_test_clean.csv")
Y_train = read.csv("~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/Y_train_clean.csv")
Y_test = read.csv("~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/Y_test_clean.csv")

X_train = X_train[,-1]
Y_train = Y_train[,-1]
X_test = X_test[,-1]
X_train_matrix = model.matrix(~., X_train)
Y_train_matrix = model.matrix(~., Y_train)
X_test_matrix = model.matrix(~., X_test)
Y_train_matrix = Y_train_matrix[,-1]

prediction_matrix = matrix(NA,nrow = nrow(X_test), ncol = ncol(Y_train))
for (i in 1:dim(Y_train)[2]) {
  print(i)
  na_index <- is.na(Y_train_matrix[,i])
  cv.fit = cv.glmnet(x = X_train_matrix[!na_index, ], y = Y_train_matrix[!na_index,i], alpha = 0)
  fit = glmnet(x = X_train_matrix[!na_index, ], y = Y_train_matrix[!na_index,i], alpha = 0,lambda = cv.fit$lambda.1se)
  pred = predict(fit,newx = X_test_matrix)  
  prediction_matrix[,i] = pred
}

final_prediction = get_matrix(prediction_matrix,Y_test)
str(final_prediction)
write.csv(final_prediction,"kaggle1.csv",row.names = FALSE)

