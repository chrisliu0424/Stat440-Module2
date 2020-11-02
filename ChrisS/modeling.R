library(h2o)
y_names = names(Ytrain[,-1])[1]
x_names = names(Xtrain[,-1])
folds = 10
seed = 5346342
n_trees = 75

full_train = cbind(Xtrain[,-1], Ytrain[,-1])

ind = sample.int(round(nrow(Xtrain)*.8))
full_valid = full_train[-ind,]
full_train = full_train[ind,]



h2o.init()

	full_train_h2o = as.h2o(full_train)
	full_valid_h2o = as.h2o(full_valid)


models = predictY(
			full_train_h2o,
			full_valid_h2o,
			x_names, y_names,
			folds, seed,
			n_trees = n_trees
)

y = as.numeric(full_valid[,y_names])

yhat_glm = as.numeric(as.data.frame(h2o.predict(models$glm, full_valid_h2o))$predict)
yhat_rf = as.numeric(as.data.frame(h2o.predict(models$rf, full_valid_h2o))$predict)
yhat_en = as.numeric(as.data.frame(h2o.predict(models$en, full_valid_h2o))$predict)
