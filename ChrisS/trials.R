library(h2o)
source('read_data.R')
source('functions.R')

valid_fraction = 0.75

y_names = names(Ytrain[,-1])[1]
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

yhats = list()
residuals = list()
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
	
	yhats[[i]] = as.data.frame(matrix(ncol = length(y_names), nrow = nrow(full_valid)))
	residuals[[i]] = as.data.frame(matrix(ncol = length(y_names), nrow = nrow(full_valid)))
	names(yhats)[i] = y_names[i]
	names(residuals)[i] = y_names[i]
	for(j in 1:length(models)){
		yhats[[i]][,j] = as.numeric(as.data.frame(h2o.predict(models[[j]], full_valid_h2o))$predict)
		residuals[[i]][,j] = yhats[[i]][,j] - full_valid[,y_names[i]]
		names(yhats[[i]]) = names(models)
		names(residuals[[i]]) = names(models)
	}
}


MSPEs = as.data.frame(matrix(ncol = length(models), nrow = length(y_names)))
names(MSPEs) = names(models)
row.names(MSPEs) = y_names
for(k in 1:length(y_names)){
	for(l in 1:length(models)){
		MSPEs[k,l] = getMSPE(full_valid[,y_names[k]], yhats[[k]][,l])
	}
}
