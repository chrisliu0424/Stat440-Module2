# Set trunc to 0 if you don't want the data truncated
trunc = 1


# Read in the cleaned data
Xtrain = read.csv('data/X_train_clean.csv')
Ytrain = read.csv('data/Y_train_clean.csv')
Xtest = read.csv('data/X_test_clean.csv')
Ytest_prompt = read.csv('data/Ytest.txt')


if(trunc == 1){
	Xtrain = head(Xtrain,100)
	Ytrain = head(Ytrain,100)
	Xtest = head(Xtest,100)
	Ytest_prompt = head(Ytest_prompt)
}
