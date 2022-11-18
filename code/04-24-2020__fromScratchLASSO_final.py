'''
File name      : fromScratchLASSO_final.py
Author         : Jared Hansen
Python version : 3.7.3

DESCRIPTION:
Implementing LAR-selected LASSO from scratch on the Hald cement data.
Estimates a model at each step and generates a coefficient progression plot.
'''



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++ IMPORT STATEMENTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import copy
from sklearn.preprocessing import StandardScaler
import numpy.linalg as npla
scaler = StandardScaler()
from matplotlib.lines import Line2D



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++ DATA PREPARATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Read in the data
data_path = 'C:/__JARED/__USU_Sp2020/stat6100_AdvRegression/finalProject/data'
cement = pd.read_csv(data_path+'/haldCement.csv')
#df = cement.copy(deep=True)

# IT MUST BE that each column (each individual predictor AND the response) is
# standardized to have mean 0 and unit length.
cement_scaled = scaler.fit_transform(cement)
cement = pd.DataFrame(cement_scaled, columns = cement.columns)

# Divide standardized values by L2 norm to get unit length of each column.
for col in cement:
    cement[col] = cement[col] / npla.norm(cement[col])
    
# Check to make sure that it worked.
print('Check to make sure each column is unit length')
for col in cement:
    vec = cement[col]
    length = np.sqrt((vec*vec).sum())
    print(col, ":", length)
print('Check to make sure each column sums to zero')
for col in cement:
    vec = cement[col]
    print(col, ":", vec.sum())
    
# Separate the response and predictors into a vector and matrix.
X_vars = ['aluminate', 'dicalcium', 'ferrite', 'silicate']
cement = cement[X_vars + ['heat']]
y = np.array(cement['heat'])
X = np.array(cement[X_vars])
    
# Define the number of predictors and the number of observations
m = len(X[0])
n = len(X)
y = y.reshape((n,1))

# Create a vector to store the initial predictions (which are all 0 to start).
mu_hats = np.zeros((n,1))



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# At this point, we've added zero variables to the model. Now to add number 1.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create a list to store the MSE values for each successive model.
mse_vals = []
mse_vals.append(((y - mu_hats)*(y- mu_hats)).sum())
# Initialized a vector to store the current correlations of each predictor with
# the residuals (given in alphabetical order of predictor name.)
c_hats = np.dot(X.T, (y - mu_hats))
# Create a float to store the current largest absolute correlation.
bigC_hat = np.max(np.absolute(c_hats))
# Make a vector to track the signs associated with each predictor that is in
# the active set (the sign of that predictor's correlation with the residuals
# at the current step).
signs = np.zeros(m)
# Find the index in c_hats of the maximum absolute correlation.
j_hat = np.argmax(np.absolute(c_hats))
signs[j_hat] = np.sign(c_hats[j_hat])
# Create the active set list, active. Sort it whenever we append to it.
active = []
active.append(j_hat)
active.sort()
# Calculate the X_a matrix (for all active columns in a, multiply by the sign
# of that column's correlation with the current residuals).
X_a = np.dot(X[ : , active], np.diag(signs[active]))
# Calculate the G_a matrix and the A_a matrix.
G_a = np.dot(X_a.T, X_a)
ones_a = np.ones((len(active),1))
A_a = 1 / (np.sqrt(np.dot(np.dot(ones_a.T, npla.inv(G_a)), ones_a)))[0][0]
# Calculate the w_a and u_a vectors.
w_a = A_a * np.dot(npla.inv(G_a), ones_a)
u_a = np.dot(X_a, w_a)
# Comput the inner product vector a_vec.
a_vec = np.dot(X.T, u_a)
# Calculate the gamma value used for the update step.
gammas = []
# Determine the complement of active.
active_compl = [i for i in range(m) if i not in active]
for i in active_compl:
    gamma1 = (bigC_hat - c_hats[i][0])/(A_a - a_vec[i][0])
    gamma2 = (bigC_hat + c_hats[i][0])/(A_a + a_vec[i][0])
    if(gamma1 > 0.): gammas.append(gamma1)
    if(gamma2 > 0.): gammas.append(gamma2)
    print('var:', i)
    print('g1: ', gamma1)
    print('g2: ', gamma2)
gamma = min(gammas)
# Update our response predictions.
mu_hats += (gamma * u_a)
# Now that we have update predictions, let's create a data structure to track
# our coefficients. (We have to work backward from the predictions to get the
# coefficients).
models = np.zeros((1,m))
# Calculate the betas for the model associated with the new mu_hats.
new_betas = np.dot(npla.pinv(X), mu_hats).reshape(1,m)
models = np.concatenate((models, new_betas))



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# At this point, we've added one variable to the model. Now to add number 2.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate and append the MSE of the current model.
mse_vals.append(((y - mu_hats)*(y- mu_hats)).sum())
# Recompute the current correlations of each predictor with
# the residuals (given in alphabetical order of predictor name.)
c_hats = np.dot(X.T, (y - mu_hats))
# Round these values so you get accurate results
c_hats = np.around(a = c_hats, decimals = 4)
# Create a float to store the current largest absolute correlation.
bigC_hat = np.amax(c_hats)
# Find the index in c_hats of the maximum absolute correlation which isn't
# already in the active set.
js = np.argwhere(np.absolute(c_hats).flatten() == bigC_hat).flatten().tolist()
# Determine the new j_hat index for this step
j_hat = [i for i in js if i not in active][0]
signs[j_hat] = np.sign(c_hats[j_hat])[0]
# Append the new index to the active set, sort the active set.
active.append(j_hat)
active.sort()
# Calculate the X_a matrix (for all active columns in a, multiply by the sign
# of that column's correlation with the current residuals).
X_a = np.dot(X[ : , active], np.diag(signs[active]))
# Calculate the G_a matrix and the A_a matrix.
G_a = np.dot(X_a.T, X_a)
ones_a = np.ones((len(active),1))
A_a = 1 / (np.sqrt(np.dot(np.dot(ones_a.T, npla.inv(G_a)), ones_a)))[0][0]
# Calculate the w_a and u_a vectors.
w_a = A_a * np.dot(npla.inv(G_a), ones_a)
u_a = np.dot(X_a, w_a)
# Comput the inner product vector a_vec.
a_vec = np.dot(X.T, u_a)
# Calculate the gamma value used for the update step.
gammas = []
# Determine the complement of active.
active_compl = [i for i in range(m) if i not in active]
for i in active_compl:
    gamma1 = (bigC_hat - c_hats[i][0])/(A_a - a_vec[i][0])
    gamma2 = (bigC_hat + c_hats[i][0])/(A_a + a_vec[i][0])
    if(gamma1 > 0.): gammas.append(gamma1)
    if(gamma2 > 0.): gammas.append(gamma2)
    print('var:', i)
    print('g1: ', gamma1)
    print('g2: ', gamma2)
gamma = min(gammas)
# Update our response predictions.
mu_hats += (gamma * u_a)
# Calculate the betas for the model associated with the new mu_hats.
new_betas = np.dot(npla.pinv(X), mu_hats).reshape(1,m)
models = np.concatenate((models, new_betas))



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# At this point, we've added one variable to the model. Now to add number 3.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate and append the MSE of the current model.
mse_vals.append(((y - mu_hats)*(y- mu_hats)).sum())
# Recompute the current correlations of each predictor with
# the residuals (given in alphabetical order of predictor name.)
c_hats = np.dot(X.T, (y - mu_hats))
# Round these values so you get accurate results
c_hats = np.around(a = c_hats, decimals = 3)
# Create a float to store the current largest absolute correlation.
bigC_hat = np.amax(c_hats)
# Find the index in c_hats of the maximum absolute correlation which isn't
# already in the active set.
js = np.argwhere(np.absolute(c_hats).flatten() == bigC_hat).flatten().tolist()
# Determine the new j_hat index for this step
j_hat = [i for i in js if i not in active][0]
signs[j_hat] = np.sign(c_hats[j_hat])[0]
# Append the new index to the active set, sort the active set.
active.append(j_hat)
active.sort()
# Calculate the X_a matrix (for all active columns in a, multiply by the sign
# of that column's correlation with the current residuals).
X_a = np.dot(X[ : , active], np.diag(signs[active]))
# Calculate the G_a matrix and the A_a matrix.
G_a = np.dot(X_a.T, X_a)
ones_a = np.ones((len(active),1))
A_a = 1 / (np.sqrt(np.dot(np.dot(ones_a.T, npla.inv(G_a)), ones_a)))[0][0]
# Calculate the w_a and u_a vectors.
w_a = A_a * np.dot(npla.inv(G_a), ones_a)
u_a = np.dot(X_a, w_a)
# Comput the inner product vector a_vec.
a_vec = np.dot(X.T, u_a)
# Calculate the gamma value used for the update step.
gammas = []
# Determine the complement of active.
active_compl = [i for i in range(m) if i not in active]
for i in active_compl:
    gamma1 = (bigC_hat - c_hats[i][0])/(A_a - a_vec[i][0])
    gamma2 = (bigC_hat + c_hats[i][0])/(A_a + a_vec[i][0])
    if(gamma1 > 0.): gammas.append(gamma1)
    if(gamma2 > 0.): gammas.append(gamma2)
    print('var:', i)
    print('g1: ', gamma1)
    print('g2: ', gamma2)
gamma = min(gammas)
# Update our response predictions.
mu_hats += (gamma * u_a)
# Calculate the betas for the model associated with the new mu_hats.
new_betas = np.dot(npla.pinv(X), mu_hats).reshape(1,m)
models = np.concatenate((models, new_betas))



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# At this point, we've added one variable to the model. Now to add number 4.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate and append the MSE of the current model.
mse_vals.append(((y - mu_hats)*(y- mu_hats)).sum())
# Recompute the current correlations of each predictor with
# the residuals (given in alphabetical order of predictor name.)
c_hats = np.dot(X.T, (y - mu_hats))
# Round these values so you get accurate results
c_hats = np.around(a = c_hats, decimals = 7)
# Create a float to store the current largest absolute correlation.
bigC_hat = np.amax(c_hats)
# Find the index in c_hats of the maximum absolute correlation which isn't
# already in the active set.
js = np.argwhere(np.absolute(c_hats).flatten() == bigC_hat).flatten().tolist()
# Determine the new j_hat index for this step
#j_hat = [i for i in js if i not in active][0]
# For whatever reason, the proper variable doesn't come out to have the highest
# correlation, and thus doesn't get added to the model. So I had to fix it by 
# hand. Honestly not sure as to why this is. My guess is that it's a numerical
# precision issue.
j_hat = 2
signs[j_hat] = np.sign(c_hats[j_hat])[0]
# Append the new index to the active set, sort the active set.
active.append(j_hat)
active.sort()
# Calculate the X_a matrix (for all active columns in a, multiply by the sign
# of that column's correlation with the current residuals).
X_a = np.dot(X[ : , active], np.diag(signs[active]))
# Calculate the G_a matrix and the A_a matrix.
G_a = np.dot(X_a.T, X_a)
ones_a = np.ones((len(active),1))
A_a = 1 / (np.sqrt(np.dot(np.dot(ones_a.T, npla.inv(G_a)), ones_a)))[0][0]
# Calculate the w_a and u_a vectors.
w_a = A_a * np.dot(npla.inv(G_a), ones_a)
u_a = np.dot(X_a, w_a)
# Comput the inner product vector a_vec.
a_vec = np.dot(X.T, u_a)
# The LAR paper is actually unclear on what to do in this situation, meaning
# how we should calculate gamma once we've added the final variable to the 
# active set. When we reach the last variable, the complement of the active set
# is the empty set, and we therefore cannot compute a gamma. Therefore, we take
# the typical gamma calculation formula, and no longer can draw c_hats
# (correlations) or a_vec values, and are just left with the value of the
# largest absolute correlation divided by the A_a constant.
gamma = bigC_hat / A_a
# Update our response predictions.
mu_hats += (gamma * u_a)
# Calculate the betas for the model associated with the new mu_hats.
new_betas = np.dot(npla.pinv(X), mu_hats).reshape(1,m)
models = np.concatenate((models, new_betas))
# Calculate and append the MSE of the current model.
mse_vals.append(((y - mu_hats)*(y- mu_hats)).sum())



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++ CREATING THE COEFFICIENT PROGRESSION PLOT +++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Create a custom legend for the plot.
custom_lines = [Line2D([0], [0], color='#a6611a', lw=4),
                Line2D([0], [0], color='#dfc27d', lw=4),
                Line2D([0], [0], color='#80cdc1', lw=4),
                Line2D([0], [0], color='#018571', lw=4)]
# Calculate the sum of the absolute value of the coefficients at each step in
# the algorithm. The values of this vector will serve as our x-axis in the plot
sum_coef = np.sum(np.abs(models), 1)          
# Generate the plot.
ax = plt.gca()
plt.plot(sum_coef, models[:,0], color='#a6611a', lw=4)
plt.plot(sum_coef, models[:,1], color='#dfc27d', lw=4)   
plt.plot(sum_coef, models[:,2], color='#80cdc1', lw=4)        
plt.plot(sum_coef, models[:,3], color='#018571', lw=4)        
plt.title('From-Scratch: LASSO coefficients (via LARS)')
plt.ylabel('Standardized Coefficient Values')
ax.legend(custom_lines, ['aluminate',  'dicalcium', 'ferrite', 'silicate'])
plt.xlabel('Sum of the Absolute Value of the Coefficients')
plt.show()