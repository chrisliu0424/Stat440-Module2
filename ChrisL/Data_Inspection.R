rm(list = ls())
library(stringr)
library(rgl)
library(dplyr)
setwd("~/Documents/GitHub/Stat440-Module2/ChrisL/Data_Original/")
X_train = read.table('~/Documents/GitHub/Stat440-Module2/ChrisL/Data_Original/Xtrain.txt',sep = ' ',header = TRUE)
X_test = read.table('~/Documents/GitHub/Stat440-Module2/ChrisL/Data_Original/Xtest.txt',sep = ' ',header = TRUE)
Y_train = read.table('~/Documents/GitHub/Stat440-Module2/ChrisL/Data_Original/Ytrain.txt',sep = ' ',header = TRUE)
Y_test = read.table('~/Documents/GitHub/Stat440-Module2/ChrisL/Data_Original/Ytest.txt',sep = ',',header = TRUE)


get_matrix = function(result,Ytest){
  # This function automately get prediction result from the corresponding column in the full prediciton dataset
  # Input: A 50000*(14+1) prediction matrix and the original Ytest dataset
  # Output: A 50000*2 prediction matrix
  # Required library: stringr
  prediction_df = Ytest
  value = strsplit(prediction_df$Id,":")
  prediction_df$column = as.numeric(str_extract(sapply(value,'[',2),'\\d+'))
  for (i in 1:nrow(Ytest)) {
    prediction_df[i,'Value'] = result[i, prediction_df$column[i]]
  }
  prediction_df$column = NULL
  return(prediction_df)
}

# Column B15 is character but it should be numeric,
# Fill NaN with median in each row
# If we don't fill NaN, filter() will filter out NaN. We will get 63,235 rows at the end(start with 153,287).
# If we use fill NaN, we will have 147,006 rows at the end.
# But filling the function using median will bump up the count of the median bar(in histogram) quiet a lot.
X_train_filled = X_train %>% mutate(B15 = as.numeric(B15)) %>% mutate_all(funs(ifelse(is.na(.),median(., na.rm = TRUE),.)))
X_test_filled = X_test %>% mutate(B15 = as.numeric(B15))

# 1st visualization
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}

# After some investigation, we can clearly see that all columns come from some distribution(Normal or Uniform mostly, some indicator function)
# Since the data is simulated, we can safely delete some extreme outliers
# After some data inspection, we can clearly see that the "EXTREME OUTLIERS" are "PERFECT NUMBERS"(some are 9)
# TODO: May choose to delete or change by median, but I will delete the observations first, since there're so many data

# 1.Filter out A01 < -5, and A01 > 5
filter(X_train_filled, A01 < (-5))
filter(X_train_filled, A01 > (5))    ####################### This is very strange #################
X_train_modified = filter(X_train_filled, A01 > (-5), A01 < 5)
# Clear all plots
dev.off(dev.list()["RStudioGD"])
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train_modified[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}


# 2.Filter out A02 < -5
filter(X_train_modified, A02 < (-5))
X_train_modified2 = filter(X_train_modified, A02 > (-5))
# Clear all plots
dev.off(dev.list()["RStudioGD"])
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train_modified2[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}


# 3.Filter out B01 < -5
filter(X_train_modified2,B01 < (-5))
X_train_modified3 = filter(X_train_modified2,B01 > (-5))
# Clear all plots
dev.off(dev.list()["RStudioGD"])
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train_modified3[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}

# 4.Filter out B02 < -5
filter(X_train_modified3,B02<(-5))
X_train_modified4 = filter(X_train_modified3,B02 > (-5))
# Clear all plots
dev.off(dev.list()["RStudioGD"])
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train_modified4[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}

# 5.Filter out B03 < -5
filter(X_train_modified4,B03<(-5))
X_train_modified5 = filter(X_train_modified4,B03>(-5))
# Clear all plots
dev.off(dev.list()["RStudioGD"])
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train_modified5[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}

# 6.Filter out B04 < -5
filter(X_train_modified5,B04<(-5))
X_train_modified6 = filter(X_train_modified5,B04>(-5))
# Clear all plots
dev.off(dev.list()["RStudioGD"])
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train_modified6[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}

# 7.Filter out B05 < -5
filter(X_train_modified6,B05 < (-5))
X_train_modified7 = filter(X_train_modified6,B05>(-5))
# Clear all plots
dev.off(dev.list()["RStudioGD"])
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train_modified7[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}

# 8.Filter out B06 < -5
filter(X_train_modified7,B06< (-5))
X_train_modified8 = filter(X_train_modified7,B06>(-5))
# Clear all plots
dev.off(dev.list()["RStudioGD"])
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train_modified8[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}

# 9.Filter out B07 < -5
filter(X_train_modified8,B07 < (-5))
X_train_modified9 = filter(X_train_modified8,B07>(-5))
# Clear all plots
dev.off(dev.list()["RStudioGD"])
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train_modified9[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}

# 10.Filter out B08 < -5
filter(X_train_modified9,B08 < (-5))
X_train_modified10 = filter(X_train_modified9,B08>(-5))
# Clear all plots
dev.off(dev.list()["RStudioGD"])
for (i in 1:ncol(X_train)) {
  hist(as.numeric(X_train_modified10[,i]),main = paste0("variable ",colnames(X_train)[i]))  
}

# Set the final Training set as X_train_final
X_train_final = X_train_modified10 %>% select(-F09,-F10,-F11,-F12,-G01,-G02,-G03)

# Getting the Y_train the same dimension as X_train
Y_train_final = Y_train[Y_train$Id %in% X_train_final$Id,] 


# Deal with the testing set
# We will fillna by the median of the corresponding columns in the training set
# We will replace extrame value with the median of the training setm
train_medians = apply(X_train_final,2,median)
# means = apply(X_train_final,2,mean)     #### I think median is better because they are very similar, but the last few predictors are indicator variable(0,1)

# We will delete the last 7 columns by now, because it's -9 everywhere in the X_test. (F09,F10,F11,F12,G01,G02,G03)
X_test_modified = select(X_test_filled,-F09,-F10,-F11,-F12,-G01,-G02,-G03)

# Fill na with the train medians 
for (i in 1:ncol(X_test_modified)) {
  X_test_modified[is.na(X_test_modified[,i]),i] = train_medians[i]  
}

# Now replace -9 nad 9 in every columns by the median of that column
for (i in 2:ncol(X_test_modified)) {
  X_test_modified[X_test_modified[,i]<(-5),i] = train_medians[i]
  X_test_modified[X_test_modified[,i]>5,i] = train_medians[i]
}

X_test_final = X_test_modified
str(X_test_final)

##### TODO: F03, F04, F05 is very strange, use table to examine
table(X_train_modified10$F03)
table(X_train_modified10$F04)
table(X_train_modified10$F05)

write.csv(X_train_final,"~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/X_train_clean.csv",row.names = FALSE)
write.csv(X_test_final,"~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/X_test_clean.csv",row.names = FALSE)
write.csv(Y_train_final,"~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/Y_train_clean.csv",row.names = FALSE)
write.csv(Y_test,"~/Documents/GitHub/Stat440-Module2/ChrisL/Data_cleaned/Y_test_clean.csv",row.names = FALSE)
