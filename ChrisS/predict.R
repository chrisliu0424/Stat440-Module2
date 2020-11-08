library(h2o)
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
Xtest_h2o = as.h2o(Xtest)


yhats = as.data.frame(matrix(ncol = 1, nrow = nrow(Xtest)))
names(yhats) = "Id"
yhats$Id = Xtest$Id

all_yhats = list()

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

	print("Predicting ...")
	for(j in 1:length(models)){
		yhats = cbind(yhats, makeZPredictions(models[[j]], Xtest_h2o))
	}
	names(yhats)[2:(1+length(models))] = names(models)
	yhats$mean_ensemble = rowMeans(yhats[,names(models)])

	# Get separate matrices formed
	all_yhats[[i]] = yhats
	names(all_yhats)[i] = y_names[i]
}
all_mats = list()
for(i in 1:length())


print("Writing Predictions")
for(i in 1:length(all_yhats))

h2o.shutdown()
