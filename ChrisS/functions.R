# Enter training and validation sets that include both
# the X and Y sets and other parameters
predictY = function(
			full_train_h2o, full_valid_h2o,
			x_names, y_names,
			folds, seed,
			n_trees
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
		family = "AUTO",
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
	en = h2o.stackedEnsemble(
		x = x_names,
		y = y_names,
		training_frame = full_train_h2o,
		validation_frame = full_valid_h2o,
		base_models = list(glm, rf, gbm)
	)
	return(list(glm = glm,rf = rf,gbm = gbm,en = en))
}

getMSPE = function(y, y_hat){
	return(sum((y - y_hat)^2))
}
