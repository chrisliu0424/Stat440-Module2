rf = read.csv('all_predictions/predictions_rf.csv')
dl = read.csv('all_predictions/predictions_dl.csv')
gbm = read.csv('all_predictions/predictions_gbm.csv')
glm = read.csv('all_predictions/predictions_glm.csv')

yhat = cbind(dl,glm$Value, gbm$Value, rf$Value)
names(yhat) = c('Id','dl','glm','gbm','rf')

ensemble = as.data.frame(matrix(ncol = 2, nrow = nrow(yhat)))
names(ensemble) = c('Id','Value')
ensemble$Id = yhat$Id

choose = c(
#	   'dl',
#	   'glm',
	   'gbm',
	   'rf'
)


for(i in 1:nrow(yhat)){
	ensemble$Value[i] = mean(as.numeric(yhat[i,choose]))
}

write.csv(	ensemble,
	  	paste0(length(choose),'ensemble.csv'),
		quote = FALSE,
		row.names = FALSE
)
