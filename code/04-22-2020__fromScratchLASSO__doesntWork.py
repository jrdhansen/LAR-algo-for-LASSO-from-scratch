'''
File name      : fromScratchLASSO.py
Author         : Jared Hansen
Python version : 3.7.3

DESCRIPTION:
Coding up LAR-selected LASSO from scratch.
             
'''



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++ IMPORT STATEMENTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


# Read in the data
data_path = 'C:/__JARED/__USU_Sp2020/stat6100_AdvRegression/finalProject/data'
cement = pd.read_csv(data_path+'/haldCement.csv')
df = cement.copy(deep=True)




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++ GLOBAL CONSTANTS, CLASS & FUNCTION DEFINITIONS ++++++++++++++++++++++++++
#++++ NOTE: these functions should be kept in this order !!!
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def calc_mean(col):
    """ This function returns the mean of col (a column from a dataframe).
    """
    sum_obs = 0
    for i in range(len(col)): 
        sum_obs += col[i]
    return(sum_obs / len(col)) 

def calc_stDev(col, col_mean):
    """ This function returns the population standard deviation of col
    (which is a column from a dataframe).
    """
    numerator = 0
    for i in range(len(col)):
        numerator += ((col[i] - col_mean) * (col[i] - col_mean))
    col_stDev = np.sqrt(numerator / (len(col)))
    return(col_stDev)

def calc_euclVecLen(col):
    """ This function calculates the Euclidean length of a column for scaling.
    """
    return(np.sqrt((col*col).sum()))

def stdz_oneVal(val, mean, stDev):
    """ This function returns a standardized single value (val), and is also
    scaled (divided by euclLen) such that each column is unit length.
    """
    return((val - mean)/(stDev))


def stdz_col(col):
    """ This function returns the standardized version of df['col'] where each
    column is unit length.
    """
    col_mean = calc_mean(col)
    col_stDev = calc_stDev(col, col_mean)
    new_col = col.apply(stdz_oneVal, mean = col_mean, stDev = col_stDev)
    col_euclLen = calc_euclVecLen(new_col)
    new_col = new_col / col_euclLen
    #df = df.drop(col, axis = 1)
    return(new_col)

def corr(col1, col2):
    """ This function returns the correlation between two columns.
    """
    products = col1 * col2
    return(products.sum() / len(col1))









#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++ STEP 1 FROM ESL BOOK, PAGE 74, LAR ALGORITHM ++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Standardize the predictors to have mean zero and unit norm.
# Start with the residual r = y - yBar and [beta1,beta2,...,betaP] = [0,...,0]
# I also standardize the response in this manner. Another minor change: I set
# the initial residuals equal to the response values, since a model with all 0
# beta coefficients simply produces all estimates as 0, making the errors just
# the original response values.



# Apply the stdz_col function to all columns of the df.
cement = df.apply(stdz_col, axis = 0)
print('Check to make sure each column is unit length')
for col in cement:
    vec = cement[col]
    length = np.sqrt((vec*vec).sum())
    print(col, ":", length)
print('Check to make sure each column sums to zero')
for col in cement:
    vec = cement[col]
    print(col, ":", vec.sum())


# Store the response values in a separate numpy array, drop the response from
# the dataframe of predictors.
response_col = 'heat'
df_cols = list(cement)
df_cols.remove(response_col)
response = np.array(cement[response_col])
response = response.reshape(len(response), 1)
cement = cement.drop(response_col, axis = 1)
# Determine the number of features for this dataset.
num_feats = len(df_cols)
# Intialize the residuals to the values of the response (this is true at the 
# outset when all beta coefficients are 0, leading to all response predictions
# being 0).
resids = copy.deepcopy(response)
# Make a copy of the cement (predictors) dataframe and use that for calculation
x_mat = cement.copy(deep=True)















#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++ STEP 2 FROM ESL BOOK, PAGE 74, LAR ALGORITHM ++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Find the predictor x_j most correlated with residuals.
corrs={}
for col in df_cols:
    current_col = np.array(x_mat[col]).reshape(len(x_mat), 1)
    corrs[col] = corr(current_col, response)


















# Move beta_j from 0 towards its least-squares coefficient <x_j, r> until some
# other competitor x_k has as much correlation with the current residual as
# does x_j.


# Move beta_j and beta_k in the direction defined by their joint least squares
# coefficient of the current residual on (x_j, x_k) until some other competitor
# x_l has as much correlation with the current residual as do (x_j and x_k).
# LASSO MOD: if a non-zero coefficient hits zero, drop its variable from the
#            active set of variables and recompute the current joint least
#            squares direction.


# Continue in this way until we arrive at the OLS beta estimates.












