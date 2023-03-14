library(glmnet)
library(dplyr)

X=read.csv("C:\\Users\\Ranjith James\\Documents\\PYMC_TRIAL\\X_P.csv")
y=read.csv("C:\\Users\\Ranjith James\\Documents\\PYMC_TRIAL\\y_P.csv")
df=read.csv("C:\\Users\\Ranjith James\\Documents\\PYMC_TRIAL\\df_P.csv")

x=as.matrix(X[,2:11])
y=as.matrix(y[,2])

lower =c(-Inf,-1.25,-Inf,-Inf,-Inf,-Inf,-Inf,0,-3.5,-Inf)
upper =c(Inf,0,Inf,Inf,Inf,Inf,Inf,2,0,Inf)

fit1=glmnet(x,y,intercept = TRUE,lower.limits =lower,upper.limits = upper )


modelcoef<-coef(object = fit1,s = 0.005) %>%as.matrix() %>%as.data.frame()
modelcoef$Variable<-row.names(modelcoef)
names(modelcoef)<-c("Coeff", "Variable")
row.names(modelcoef)<-1:nrow(modelcoef)
