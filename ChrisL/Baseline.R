source("~/Documents/GitHub/Stat440-Module2/ChrisL/functions.R")

rm(list = ls())
Y_train = read.table('~/Documents/GitHub/Stat440-Module2/ChrisL/Data_Original/Ytrain.txt',sep = ' ',header = TRUE)
Y_test = read.table('~/Documents/GitHub/Stat440-Module2/ChrisL/Data_Original/Ytest.txt',sep = ',',header = TRUE)

Y_train = Y_train[,-1]
col_mean = apply(Y_train,2,mean,na.rm = TRUE)

baseline_matrix = matrix(NA,nrow = 50000,ncol = 14)

for (i in 1:14) {
  baseline_matrix[,i] = col_mean[i]
}

