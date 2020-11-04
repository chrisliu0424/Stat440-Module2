# Set trunc to 0 if you don't want the data truncated
trunc = 100
print(paste('Trunc:',as.character(trunc)))


# Read in the cleaned data
print("Reading Comma Separated Files")
Xtrain = read.csv('data/Xtrain.csv', nrows = trunc)
Ytrain = read.csv('data/Ytrain.csv', nrows = trunc)
Xtest = read.csv('data/Xtest.csv')
Ytest_prompt = read.csv('data/Ytest.txt')
Ytest_prompt$Value = as.numeric(Ytest_prompt$Value)

print(paste('Dim Xtrain:',as.character(dim(Xtrain)[1]),'x',as.character(dim(Xtrain)[2])))
print(paste('Dim Ytrain:',as.character(dim(Ytrain)[1]),'x',as.character(dim(Ytrain)[2])))
print(paste('Dim Xtest:',as.character(dim(Xtest)[1]),'x',as.character(dim(Xtest)[2])))
print(paste('Dim Ytest_prompt:',as.character(dim(Ytest_prompt)[1]),'x',as.character(dim(Ytest_prompt)[2])))


