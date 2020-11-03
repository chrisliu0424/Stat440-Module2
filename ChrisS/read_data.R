# Set trunc to 0 if you don't want the data truncated
trunc = 0


# Read in the cleaned data
print("Reading Comma Separated Files")
Xtrain = read.csv('data/Xtrain.csv')
Ytrain = read.csv('data/Ytrain.csv')
Xtest = read.csv('data/Xtest.csv')
Ytest_prompt = read.csv('data/Ytest.txt')

print(paste('Dim Xtrain:',as.character(dim(Xtrain))))
print(paste('Dim Ytrain:',as.character(dim(Ytrain))))
print(paste('Dim Xtest:',as.character(dim(Xtest))))
print(paste('Dim Ytest_prompt:',as.character(dim(Ytest_prompt))))

print(paste('Trunc:',as.character(trunc)))

if(trunc == 1){
	Xtrain = head(Xtrain,100)
	Ytrain = head(Ytrain,100)
	Xtest = head(Xtest,100)
}


print(paste('Dim Xtrain:',as.character(dim(Xtrain))))
print(paste('Dim Ytrain:',as.character(dim(Ytrain))))
print(paste('Dim Xtest:',as.character(dim(Xtest))))
print(paste('Dim Ytest_prompt:',as.character(dim(Ytest_prompt))))
