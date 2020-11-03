library(h2o)
y_names = names(Ytrain[,-1])[1]
x_names = names(Xtrain[,-1])
folds = 10
seed = 5346342
n_trees = 200
hidden_layers = c(200,200)
# Number of epochs for the deep learning model
e = 100

full_train = cbind(Xtrain[,-1], Ytrain[,-1])

ind = sample.int(round(nrow(Xtrain)*.8))
full_valid = full_train[-ind,]
full_train = full_train[ind,]



h2o.init()

	full_train_h2o = as.h2o(full_train)
	full_valid_h2o = as.h2o(full_valid)


models = trainModels(
			full_train_h2o,
			full_valid_h2o,
			x_names, y_names,
			folds, seed,
			n_trees = n_trees,
			hidden_layers,
			n_epochs = e
)

y = as.numeric(full_valid[,y_names])

yhats = as.data.frame(matrix(ncol = length(models), nrow = length(y)))
names(yhats) = names(models)
for(i in 1:ncol(yhats)){
	yhats[,i] = as.numeric(as.data.frame(h2o.predict(models[[i]], full_valid_h2o))$predict)
}

residuals = yhats
for(i in 1:ncol(yhats)){
	for(j in 1:nrow(yhats)){
		residuals[j,i] = yhats[j,i] - y[j]
	}
}

MSPEs = data.frame(matrix(ncol = length(models), nrow = 1))
names(MSPEs) = names(models)
for(i in 1:ncol(yhats)){
	MSPEs[,i] = getMSPE(y, yhats[,i])
}
