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
	
	#######################
	### GBM Grid Search ###
	#######################
	# GBM Hyperparamters
	learn_rate_opt <- c(0.01, 0.03)
	max_depth_opt <- c(3, 4, 5, 6, 9)
	sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
	col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
	
	hyper_params <- list(learn_rate = learn_rate_opt,
	                     max_depth = max_depth_opt,
	                     sample_rate = sample_rate_opt,
	                     col_sample_rate = col_sample_rate_opt
	)

	search_criteria <- list(strategy = "RandomDiscrete",
	                        max_models = 3,
	                        seed = seed
	)

	gbm_grid <- h2o.grid(algorithm = "gbm",
	                     grid_id = "gbm_grid_binomial",
	                     x = x_names,
	                     y = y_names,
	                     training_frame = full_train_h2o,
	                     ntrees = n_trees,
	                     seed = seed,
	                     nfolds = folds,
	                     keep_cross_validation_predictions = TRUE,
	                     hyper_params = hyper_params,
	                     search_criteria = search_criteria
	)

	# Train a stacked ensemble using the GBM grid
	ensemble <- h2o.stackedEnsemble(x = x_names,
	                                y = y_names,
	                                training_frame = full_train_h2o,
	                                base_models = gbm_grid@model_ids)


#	en = h2o.stackedEnsemble(
#		x = x_names,
#		y = y_names,
#		training_frame = full_train_h2o,
#		base_models = list(glm, rf, gbm, dl)
#	)
	return(
		list(	
			glm = glm,
	            	rf = rf,
	            	gbm = gbm,
	            	dl = dl,
			ensemble = ensemble
	#           	en = en
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
