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
#	en = h2o.stackedEnsemble(
#		x = x_names,
#		y = y_names,
#		training_frame = full_train_h2o,
#		base_models = list(glm, rf, gbm, dl)
#	)
	return(list(glm = glm,
	            rf = rf,
	            gbm = gbm,
	            dl = dl
	#           en = en
	))
}

getMSPE = function(y, y_hat){
	return(mean((y - y_hat)^2))
}

# Takes in the column to predict and that
# model designated for that column, then
# returns the updates Ytest with the predicted
# values
predictIds = function(Ytest, Xtest, column, h2o_model){
	inds = grep(column,Ytest$Id)
	ids = as.numeric(gsub(':.*$','',Ytest$Id[inds]))
	Xtest = as.h2o(Xtest[which(Xtest$Id %in% ids),])
	y_hat = as.numeric(as.data.frame(h2o.predict(h2o_model, Xtest))$predict)
	Ytest$Value[inds] = y_hat
	return(Ytest)
}

# Returns the prediction vector for all indeces of the current response
makeZPredictions = function(h2o_model, Xtest){
		return(as.numeric(as.data.frame(h2o.predict(h2o_model, Xtest))$predict))
}


getPredictions = function(result,Ytest){
  # This function automately get prediction result from the corresponding column in the full prediciton dataset
  # Input: A 50000*p prediction matrix and the original Ytest dataset
  # Output: A 50000*2 prediction matrix
  # Required library: stringr
  Ytest$Id = as.character(Ytest)
  prediction_df = Ytest
  value = strsplit(prediction_df$Id,":")
  prediction_df$row = as.numeric(str_extract(sapply(value,'[',1),'\\d+'))
  prediction_df$column = as.numeric(str_extract(sapply(value,'[',2),'\\d+'))
  for (i in 1:nrow(Ytest)){
    prediction_df[i,'Value'] = result[i, prediction_df$column[i]]
  }
  prediction_df$Id = as.integer(prediction_df$row)
  prediction_df$column = NULL
  prediction_df$row = NULL
  prediction_df$Value = as.numeric(prediction_df$Value)
  return(prediction_df)
}
