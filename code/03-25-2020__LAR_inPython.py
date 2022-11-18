'''
File name      : 03-25-2020__LAR_inPython.py
Author         : Jared Hansen
Date created   : 03/25/2020
Python version : 3.7.3

DESCRIPTION:
Coding up LAR-selected LASSO from scratch
Using someone else's LAR code found here
https://github.com/hughperkins/selfstudy-LARS/blob/master/test_lars.ipynb
when I get stuck.
             

Another good site: https://statweb.stanford.edu/~tibs/lasso/simple.html
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














#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++ GLOBAL CONSTANTS, CLASS & FUNCTION DEFINITIONS ++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

















#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++ PROCEDURAL CODE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





# Read in the data
data_path = 'C:/__JARED/__USU_Sp2020/stat6100_AdvRegression/finalProject/data'
tinder = pd.read_csv(data_path+'/tinder.csv')

# Any necessary data cleaning
#------------------------------------------------------------------------------
# Drop the ID column (not a predictor, not useful for anything else).
tinder = tinder.drop(['ID'], axis = 1)
# Convert all columns to be numeric types (some are strings).
tinder = tinder.apply(pd.to_numeric, errors = 'coerce')
# How many missing values does this dataset have?  --> answer is 7
missing = tinder[tinder.isnull().any(axis = 1)]
# There are few enough rows with missing values that we'll just drop them.
tinder = tinder.dropna(axis = 0, how = 'any')
tinder = tinder.reset_index(drop=True)



df = tinder.copy(deep=True)

#--------------------------------------------------------------------
# Standardize each individual column (predictors AND response)
#--------------------------------------------------------------------

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

def stdz_oneVal(val, mean, stDev):
    """ This function returns a standardized a single value (val).
    """
    return((val - mean)/stDev)

def stdz_col(col):
    """ This function returns the standardized version of df['col'].   
    """
    col_mean = calc_mean(col)
    col_stDev = calc_stDev(col, col_mean)
    new_col = col.apply(stdz_oneVal, mean = col_mean, stDev = col_stDev)
    #df = df.drop(col, axis = 1)
    return(new_col)


# Apply the stdz_col function to all columns of the df.
tinder = df.apply(stdz_col, axis = 0)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++ AT THIS POINT WE HAVE STANDARDIZED ALL COLUMNS (VARS & RESPONSE)
#++++++ IN OUR DATASET.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++







def corr(col1, col2):
    """ This function returns the correlation between two columns.
    """
    products = col1 * col2
    return(products.sum() / len(col1))
    
"""
col1 = tinder['Genuine']
col2 = tinder['SelfEsteem']
corr(col1, col2)

pandas_corrs = df.corr()
pandas_corrs.loc['Genuine', 'SelfEsteem']
"""



# These things are done once at the beginning of calculating coefficients.
# Store the response values in a separate numpy array, drop the response from
# the dataframe of predictors.
response = 'Genuine'
df_cols = list(tinder)
df_cols.remove(response)
response = np.array(tinder['Genuine'])
response = response.reshape(len(response), 1)
tinder = tinder.drop('Genuine', axis = 1)
# Determine the number of features for this dataset.
num_feats = len(df_cols)


def calc_sbc(residuals, num_feats):
    """ This function calculates the SBC value for a given model.
    """
    # Calculate the SSE
    sse = sum(residuals**2)[0]
    # Store other values needed for computing SBC
    n = len(residuals)
    p = num_feats
    # Calculate and return the SBC
    sbc = n * np.log((sse / n)) + p * np.log(n)
    return(sbc)




# Keep track of the current coefficient estimates in a pandas dataframe.
# All coefficients are initialized to zero.
coefs = pd.DataFrame(np.zeros((1, len(df_cols))), columns = df_cols)
# Add another column which tracks SBC for each iteration of the model.
coefs['sbc'] = calc_sbc(response, len(df_cols))

# Intialize the residuals to the values of the response (this is true at the 
# outset when all beta coefficients are 0, leading to all response predictions
# being 0).
resids = copy.deepcopy(response)


step_size = 0.05
x_mat = tinder.copy(deep=True)
var_maxCorr = 'SelfEsteem'
























# TODO: did a BIG FREAKING OOPSIE. have to SIMULTANEOUSLY move all entered
# coefficients in the respective directions of their correlations with the
# residuals  !!!!



# TODO: nothing is coming into the model other than the first two variables (SelfValidation, SelfEsteem)



# Thoughts: make a running list which tracks the variables whose coefs need to
#           be adjusted at each step.

# This list tracks which variables have been entered into the model.
incl_vars = []



def adjust_coef(resids, coefs, incl_vars, step_size=0.1, num_feats=20, x_mat=x_mat):
    # Calculate and store the pairwise correlation of each predictor with the
    # residuals from the current model (we can actually skip the step of 
    # finding the variable with highest correlation with the response since the
    # initial predictions are all 0, making the residuals the response values
    # themselves).
    corrs = {}
    for feature in df_cols:
        current_col = np.array(x_mat[feature]).reshape(len(x_mat), 1)
        one_corr = corr(current_col, resids)
        corrs[feature] = one_corr
    # Compute predicted response values using current beta coefficients.
    current_betas = np.array(coefs.iloc[-1, 0:-1]).reshape(len(list(x_mat)), 1)
    y_hats = np.dot(x_mat, current_betas)
    # Calculate residuals
    resids = resids - y_hats
    
    # TODO Determine if we can enter a new variable into the model (is there a var
    # whose correlation with the residuals is at least as hight )
    
    
    # Find the variable (that isn't already included in the model) with the
    # highest correlation with the residuals .
    #unused_vars = [v for v in df_cols if v not in incl_vars]
    unusedVar_maxCorr = max(corrs, key=lambda y: abs(corrs[y]))    
    # Find the sign of the variable with highest abolute correlation
    #move_dir = np.sign(corrs[unusedVar_maxCorr])
    # Is this variable is in the model yet? If not, add it to the list.
    if(not (unusedVar_maxCorr in incl_vars) ):
        incl_vars.append(unusedVar_maxCorr)
    # Create a new row for the coefs dataframe (we're adjusting the coefficients).
    new_betas = coefs.iloc[[-1]]
    # For every variable in incl_vars, adjust their beta coefficient in the
    # direction of their correlation with the residuals
    for var in incl_vars:
        # Determine the move direction (move_dir) for this variable.
        move_dir = np.sign(corrs[var])
        # Calculate the new coefficient for this variable, save it to coefs.
        new_coef = coefs.iloc[-1][var] + (move_dir * step_size)
        new_betas[var] = new_coef
    # Caluclate the new SBC and add it to the new_betas array
    new_betas['sbc'] = calc_sbc(resids, num_feats)
    # Append the new row of coefficients to the coefs dataframe.
    coefs = coefs.append(new_betas, ignore_index = True)
    return((resids, coefs, incl_vars))


trial_run = adjust_coef(num_feats, x_mat, resids, step_size, coefs, incl_vars)
trial_run2 = adjust_coef(num_feats, x_mat, trial_run[0], step_size, trial_run[2], trial_run[3])
trial_run3 = adjust_coef(num_feats, x_mat, trial_run2[0], step_size, trial_run2[1], trial_run2[2], trial_run2[3])
trial_run4 = adjust_coef(num_feats, x_mat, trial_run3[0], step_size, trial_run3[1], trial_run3[2], trial_run3[3])
trial_run5 = adjust_coef(num_feats, x_mat, trial_run4[0], step_size, trial_run4[1], trial_run4[2], trial_run4[3])
trial_run6 = adjust_coef(num_feats, x_mat, trial_run5[0], step_size, trial_run5[1], trial_run5[2], trial_run5[3])
trial_run7 = adjust_coef(num_feats, x_mat, trial_run6[0], step_size, trial_run6[1], trial_run6[2], trial_run6[3])
trial_run8 = adjust_coef(num_feats, x_mat, trial_run7[0], step_size, trial_run7[1], trial_run7[2], trial_run7[3])
trial_run9 = adjust_coef(num_feats, x_mat, trial_run8[0], step_size, trial_run8[1], trial_run8[2], trial_run8[3])
trial_run10 = adjust_coef(num_feats, x_mat, trial_run9[0], step_size, trial_run9[1], trial_run9[2], trial_run9[3])
trial_run11 = adjust_coef(num_feats, x_mat, trial_run10[0], step_size, trial_run10[1], trial_run10[2], trial_run10[3])
trial_run12 = adjust_coef(num_feats, x_mat, trial_run11[0], step_size, trial_run11[1], trial_run11[2], trial_run11[3])
trial_run13 = adjust_coef(num_feats, x_mat, trial_run12[0], step_size, trial_run12[1], trial_run12[2], trial_run12[3])
trial_run14 = adjust_coef(num_feats, x_mat, trial_run13[0], step_size, trial_run13[1], trial_run13[2], trial_run13[3])

trial_run15 = adjust_coef(num_feats, x_mat, trial_run14[0], step_size, trial_run14[1], trial_run14[2], trial_run14[3])
trial_run16 = adjust_coef(num_feats, x_mat, trial_run15[0], step_size, trial_run15[1], trial_run15[2], trial_run15[3])
trial_run17 = adjust_coef(num_feats, x_mat, trial_run16[0], step_size, trial_run16[1], trial_run16[2], trial_run16[3])
trial_run18 = adjust_coef(num_feats, x_mat, trial_run17[0], step_size, trial_run17[1], trial_run17[2], trial_run17[3])
trial_run19 = adjust_coef(num_feats, x_mat, trial_run18[0], step_size, trial_run18[1], trial_run18[2], trial_run18[3])
trial_run20 = adjust_coef(num_feats, x_mat, trial_run19[0], step_size, trial_run19[1], trial_run19[2], trial_run19[3])
trial_run21 = adjust_coef(num_feats, x_mat, trial_run20[0], step_size, trial_run20[1], trial_run20[2], trial_run20[3])
trial_run22 = adjust_coef(num_feats, x_mat, trial_run21[0], step_size, trial_run21[1], trial_run21[2], trial_run21[3])

trial_run22 = adjust_coef(num_feats, x_mat, trial_run21[0], step_size, trial_run21[1], trial_run21[2], trial_run21[3])


trial_run = adjust_coef(resids, coefs, incl_vars)
for i in range(500):
    print(i)
    trial_run = adjust_coef(trial_run[0], trial_run[1], trial_run[2])





"""
# Calculate the correlation of all columns with the response, save as a dict.
for feature in df_cols:
    current_col = np.array(tinder[feature]).reshape(len(tinder), 1)
    one_corr = corr(current_col, response)
    print(one_corr , "  :  ", feature )
    init_corrs[feature] = one_corr
"""


# Write a function that computes the regression estimates for the training data
# y_hat = [X][beta_hat] where y_hat is the vector of predicted values, X is
# the matrix of observations (standardized) and beta_hat is our current set of
# coefficients (the furthest-down row in the ceofs dataframe).
current_betas = np.array(coefs.iloc[[-1]]).reshape(len(list(tinder)), 1)
y_hat = np.dot(tinder, current_betas)

# Calculate the residuals
resids = response - y_hat


# Compute correlations of the residuals with all un-entered predictors
resid_corrs = {}
for feature in df_cols:
    current_col = np.array(tinder[feature]).reshape(len(tinder), 1)
    one_corr = corr(current_col, resids)
    print(one_corr , "  :  ", feature )
    resid_corrs[feature] = one_corr


# Move the highst-correlated variable's beta coef in the direction of its sign.

# Find the variable with the highest (absolute) correlation with the response.
var_maxCorr = max(init_corrs, key=lambda y: abs(init_corrs[y]))

# Find the sign of the variable with highest abolute correlation
np.sign(init_corrs[var_maxCorr])




# Move the coefficient in the direction of its correlation. For each step:
    # Compute the regression estimates for the training data
    # Compute the residuals for the regression estimates
    # Compute the correlation of the residuals with all un-entered predictors



# TODO: WRITE A FUNCTION FOR COMPUTING REGRESSION ESTIMATES. The inputs would
#       be the data, coefficient values, maybe need a list of column names(?)

# At each step compute the residuals r_1 = ( y - (beta0)(xj) )
# TODO: WRITE A FUNCTION FOR COMPUTING RESIDUALS

















































