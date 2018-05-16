
# linear regression


rm(list=ls())
graphics.off()


# perfect fit
x <- seq(from=1, to=20, by=1)
y <- 2*x

my_lm <- lm(y ~ x)
summary(my_lm)

plot(x, y, pch=16, cex=1, col='blue')
points(x, my_lm$fitted.values, pch=16, cex=1, col='red')
points(x, my_lm$fitted.values, type='l', col='red')
display_text = paste("R_squared=", round(summary(my_lm)$r.squared, digits=4),
                     ", p-value=", round(summary(my_lm)$coefficients[4], digits=4),
                     sep="")
title(main=display_text)





# add some noise
y <- 2*x + rnorm(length(x), mean=0, sd=4.0)

my_lm <- lm(y ~ x)
summary(my_lm)

plot(x, y, pch=16, cex=1, col='blue', main=title)
points(x, my_lm$fitted.values, pch=16, cex=1, col='red')
points(x, my_lm$fitted.values, type='l', col='red')
display_text = paste("R_squared=", round(summary(my_lm)$r.squared, digits=4),
                     ", p-value=", round(summary(my_lm)$coefficients[4], digits=4),
                     sep="")
title(main=display_text)






# perfect line with an outlier
x <- seq(from=1, to=20, by=1)
y <- 2*x

x <- c(x, 100)
y <- c(y, 500)
plot(x, y, pch=16, cex=1, col='blue')
my_lm <- lm(y ~ x)
plot(x, y, pch=16, cex=1, col='blue', main=title)
points(x, my_lm$fitted.values, pch=16, cex=1, col='red')
points(x, my_lm$fitted.values, type='l', col='red')
display_text = paste("R_squared=", round(summary(my_lm)$r.squared, digits=4),
                     ", p-value=", round(summary(my_lm)$coefficients[4], digits=4),
                     sep="")
title(main=display_text)








# pure noise
x <- seq(from=1, to=20, by=1)
y <- rnorm(length(x), mean=0, sd=5)
plot(x, y, pch=16, cex=1, col='blue')
my_lm <- lm(y ~ x)
plot(x, y, pch=16, cex=1, col='blue')
points(x, my_lm$fitted.values, pch=16, cex=1, col='red')
points(x, my_lm$fitted.values, type='l', col='red')
display_text = paste("R_squared=", round(summary(my_lm)$r.squared, digits=4),
                     ", p-value=", round(summary(my_lm)$coefficients[4], digits=4),
                     sep="")
title(main=display_text)



# add outlier
x <- c(x, 100)
y <- c(y, 100)

plot(x, y, pch=16, cex=1, col='blue')
my_lm <- lm(y ~ x)
plot(x, y, pch=16, cex=1, col='blue', main=title)
points(x, my_lm$fitted.values, pch=16, cex=1, col='red')
points(x, my_lm$fitted.values, type='l', col='red')
display_text = paste("R_squared=", round(summary(my_lm)$r.squared, digits=4),
                     ", p-value=", round(summary(my_lm)$coefficients[4], digits=4),
                     sep="")
title(main=display_text)
