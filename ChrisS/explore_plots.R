# Script Makes plots of all variables against all responses
correlated_vars = c(	gsub('^','C0',as.character(1:3)),
			gsub('^','D0',as.character(1:9)),
			gsub('^','D',as.character(10:11)),
			gsub('^','F0',as.character(6:8))
)

for(i in 1:length(correlated_vars)){
	print(paste0('Plotting ', correlated_vars[i], ' v '))
	for(j in 2:length(names(Ytrain))){
		print(names(Ytrain)[j])
		pdf(paste0(	'explore_plots/',
				names(Ytrain)[j],
				'/',
				correlated_vars[i],
				'.pdf'
		))
		plot(
			x = Xtrain[,correlated_vars[i]],
			y = Ytrain[,names(Ytrain)[j]],
			main = paste0(correlated_vars[i],' v. ',names(Ytrain)[j]),
			ylab = names(Ytrain)[j],
			xlab = correlated_vars[i]
		)
		dev.off()
	}
}

