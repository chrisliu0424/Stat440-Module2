library(stringr)
# Enter training and validation sets that include both
# the X and Y sets and other parameters
trainModels = function(
			full_train_h2o, full_valid_h2o,
			x_names, y_names,
			folds, seed,
			n_trees, hidden_layers, n_epochs
){
	# Models
	print("GLM")
	glm = h2o.glm(
		x = x_names,
		y = y_names,
		training_frame = full_train_h2o,
		validation_frame = full_valid_h2o,
		keep_cross_validation_predictions = TRUE,
		score_each_iteration = TRUE,
		nfolds = folds,
		seed = seed,
		#family = "AUTO",
		missing_values_handling = "MeanImputation",
		remove_collinear_columns = TRUE
	)
	print("RF")
	rf = h2o.randomForest(
		x = x_names,
		y = y_names,
		training_frame = full_train_h2o,
		validation_frame = full_valid_h2o,
		keep_cross_validation_predictions = TRUE,
		score_each_iteration = TRUE,
		max_depth = 20,
		nfolds = folds,
		seed = seed,
		ntrees = n_trees
	)
	print("GBM")
	gbm = h2o.gbm(
		x = x_names,
		y = y_names,
		training_frame = full_train_h2o,
		validation_frame = full_valid_h2o,
		keep_cross_validation_predictions = TRUE,
		score_each_iteration = TRUE,
		max_depth = 20,
		nfolds = folds,
		seed = seed,
		ntrees = n_trees
	)
	print("DL")
	dl = h2o.deeplearning(
		x = x_names,
		y = y_names,
		training_frame = full_train_h2o,
		validation_frame = full_valid_h2o,
		keep_cross_validation_predictions = TRUE,
		score_each_iteration = TRUE,
		nfolds = folds,
		seed = seed,
		hidden = hidden_layers,
		epochs = n_epochs
	)
	print("XGB")
	xgb = h2o.xgboost(
		x = x_names,
		y = y_names,
		training_frame = full_train_h2o,
		validation_frame = full_valid_h2o,
		booster = "dart",
		normalize_type = "tree",
		seed = seed
	)
	print("ISO")
	iso = h2o.isolationForest(
		x = x_names,
		training_frame = full_train_h2o,
		sample_rate = 0.1,
		max_depth = 20,
		ntrees = n_trees
	)
	return(
		list(	
			glm = glm,
	            	rf = rf,
	            	gbm = gbm,
	            	dl = dl,
			xgb = xgb,
			iso = iso
		)
	)
}

getMSPE = function(y, y_hat){
	return(mean((y - y_hat)^2))
}

getZIds = function(Ytest, column){
	indeces = grep(column, Ytest$Id)
	return(list(ids = as.numeric(gsub(':.*$','',Ytest$Id[indeces])), indeces = indeces))
}

# Returns the prediction vector for all indeces of the current response
# Xtest needs to only include the rows to predict and be an h2o object
makeZPredictions = function(h2o_model, Xtest){
		return(as.numeric(as.data.frame(h2o.predict(h2o_model, Xtest))$predict))
}

meanZPredictions = function(length_ids, column, Ytrain){
	return(rep(mean(Ytrain[,column]), times = length_ids))
}
