# Script Makes plots of all variables against all responses

for(i in 2:ncol(Xtrain)){
	pdf(paste0('explore_plots/',names(Xtrain)[i])) 
}

