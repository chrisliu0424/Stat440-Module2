library(h2o)
library(rlist)
source('read_data.R')
source('functions.R')

# Parameter Assigning
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


predictions = list()
# For each response variable train a model and make predictions
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
	ids = getZIds(Ytest_prompt, y_names[i])
	indeces = ids$indeces
	ids = ids$ids
	Xtest_h2o = as.h2o(Xtest[which(Xtest$Id %in% ids),])

	print("Predicting ...")
	for(j in 1:length(models)){
		if(i == 1){
			predictions[[j]] = Ytest_prompt
		}
		predictions[[j]][indeces,'Value'] = makeZPredictions(models[[j]], Xtest_h2o)
	}
	
}
names(predictions) = names(models)
all_preds = cbind(Id = Ytest_prompt$Id, list.cbind(predictions)[,paste0(names(models),'.Value')])
predictions$means = Ytest_prompt

for(i in 1:nrow(Ytest_prompt)){
	predictions$means[i,'Value'] = mean(as.numeric(all_preds[i,2:ncol(all_preds)]))
}

for(i in 1:length(predictions)){
	predictions[[i]][,'Id'] = gsub(':.*$','',predictions[[i]][,'Id'])
}


print("Writing Predictions")
for(i in 1:length(predictions)){
	write.csv(	predictions[[i]], 
			paste0('current_predictions/',names(predictions)[i],'.csv'),
			quote = FALSE,
			row.names = FALSE
	)
}

h2o.shutdown()
