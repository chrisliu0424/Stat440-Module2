library(h2o)
source('read_data.R')
source('functions.R')


y_names = names(Ytrain[,-1])
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


all_models = list()
for(i in 1:length(unique(Ytest_prompt$y))){
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
}
names(all_models) = y_names

Ytest_prompt$Value = numeric(nrow(Ytest_prompt))
print("Predicting")
for(i in 1:nrow(Ytest_prompt)){
	Ytest_prompt$Value[i] = as.numeric(as.data.frame(
			h2o.predict(all_models[[Ytest_prompt$y[i]]]$en, as.h2o(
							Xtest[which(Xtest$Id == Ytest_prompt$Id[i])]
			))
	))
}

Ytest_prompt$Id = paste0(as.character(Ytest_prompt$Id),Ytest_prompt$y)

print('Writing Results')
Ytest_prompt = Ytest_prompt[,1:2]
print('Done')

write.csv(Ytest_prompt, "first_attempt.csv", quote = FALSE, row.names = FALSE)
