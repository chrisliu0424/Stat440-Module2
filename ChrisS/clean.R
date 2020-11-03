x = Xtest
ind = which(Xtest$B15 == "?")
x$B15[ind] = "0"
x$B15 = as.numeric(x$B15)

x$B15[ind] = mean(x$B15[-ind])


