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

Ytest_rf = Ytest_prompt
Ytest_dl = Ytest_prompt
Ytest_gbm = Ytest_prompt
Ytest_glm = Ytest_prompt
#Ytest_en = Ytest_prompt

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
	Ytest_rf = predictIds(Ytest_rf, Xtest, y_names[i], models$rf)
	Ytest_dl = predictIds(Ytest_dl, Xtest, y_names[i], models$dl)
	Ytest_gbm = predictIds(Ytest_gbm, Xtest, y_names[i], models$gbm)
	Ytest_glm = predictIds(Ytest_glm, Xtest, y_names[i], models$glm)
#	Ytest_en = predictIds(Ytest_en, Xtest, y_names[i], models$en)
	print(paste(
			as.character(nrow(Ytest_gbm) - length(which(is.na(Ytest_gbm$Value)))),
			"Total Predicted;",
			as.character(length(which(is.na(Ytest_gbm$Value)))),
			"Left to Predict"
	))

}
print("Writing Predictions")
Ytest_rf$Id = gsub(':.*$','',Ytest_rf$Id)
write.csv(Ytest_rf, "all_predictions/rf.csv", quote = FALSE, row.names = FALSE)
Ytest_dl$Id = gsub(':.*$','',Ytest_dl$Id)
write.csv(Ytest_dl, "all_predictions/dl.csv", quote = FALSE, row.names = FALSE)
Ytest_gbm$Id = gsub(':.*$','',Ytest_gbm$Id)
write.csv(Ytest_gbm, "all_predictions/gbm.csv", quote = FALSE, row.names = FALSE)
Ytest_glm$Id = gsub(':.*$','',Ytest_glm$Id)
write.csv(Ytest_glm, "all_predictions/glm.csv", quote = FALSE, row.names = FALSE)
#Ytest_en$Id = gsub(':.*$','',Ytest_en$Id)
#write.csv(Ytest_en, "predictions_en.csv", quote = FALSE, row.names = FALSE)

h2o.shutdown()
