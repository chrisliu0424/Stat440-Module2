library(h2o)
source('read_data.R')
source('functions.R')

valid_fraction = 0.75

y_names = names(Ytrain[,-1])
x_names = names(Xtrain[,-1])

folds = 10
seed = 5346342
n_trees = 200
hidden_layers = c(200,200)
# Number of epochs for the deep learning model
e = 100
 

full_train = cbind(Xtrain[,-1], Ytrain[,-1])

# Split into training and validation set
ind = head(sample.int(nrow(Xtrain)), round(nrow(Xtrain)*valid_fraction))
full_valid = full_train[-ind,]
full_train = full_train[ind,]

h2o.init()

full_train_h2o = as.h2o(full_train)
full_valid_h2o = as.h2o(full_valid)

for(i in 1:length(y_names)){
	print(paste0('Training Model for ',y_names[i]))
	models = trainModels(
				full_train_h2o,
				full_valid_h2o,
				x_names, y_names[i],
				folds, seed,
				n_trees = n_trees,
				hidden_layers,
				n_epochs = e
	)
	print(paste0(y_names[i], ' Trained'))
	print("Predicting ...")
	Ytest_prompt = predictIds(Ytest_prompt, Xtest, y_names[i], models$en)
	print(paste(
			as.character(nrow(Ytest_prompt) - length(is.na(Ytest_prompt$Value))),
			"Total Predicted;",
			as.character(length(is.na(Ytest_prompt$Value))),
			"Left to Predict"
	))
}
print("Writing Predictions")
write.csv(Ytest_prompt, "first_attempt.csv", quote = FALSE, row.names = FALSE)
