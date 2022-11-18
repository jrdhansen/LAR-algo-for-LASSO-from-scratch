#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# File name      : R_cementLASSO.R
# Author         : Jared Hansen
# R version      : 3.6.3
# 
# DESCRIPTION: fitting LASSO model to the Hald cement data, generating coef
#              progression plot, and MSE of best model (chosen by mean Cv err).
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# IMPORT STATEMENTS
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
library(glmnet)
library(ggplot2)
library(useful)
library(coefplot)
library(DT)

# Read in the data
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
path <- "C:/__JARED/__USU_Sp2020/stat6100_AdvRegression/finalProject/data/haldCement.csv"
cement <- read.table(path, header=TRUE, sep=",")
# Split into x_mat (predictor values) and y_vec (response values).
x_mat = as.matrix(cement[c('aluminate', 'ferrite', 'silicate', 'dicalcium')])
y_vec = as.matrix(cement['heat'])

# Fit LASSO using cv.glmnet. By specifying nfolds=nrow(x_mat)-1 we are 
# effectively doing PRESS (leave-one-out CV) for our selection method.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cvfit = cv.glmnet(x_mat, y_vec, nfolds=(nrow(x_mat)-1), )
plot(cvfit, xvar="lambda", label=TRUE)
coefpath(cvfit, "Hald Cement")

# Determine the MSE of the best selected model (model with lowest mean CV error)
coef(cvfit, s = "lambda.min")
preds = predict(cvfit, newx=x_mat, s="lambda.min")
mse = mean((preds - y_vec)^2)
mse