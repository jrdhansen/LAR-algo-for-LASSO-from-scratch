'''
File name      : easyLASSO.py
Author         : Jared Hansen
Python version : 3.7.3

DESCRIPTION:
Coding up LAR-selected LASSO from scratch, using lots of built-in functions.
             
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
from sklearn.preprocessing import StandardScaler
import numpy.linalg as npla
scaler = StandardScaler()
from matplotlib.lines import Line2D


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
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++ EVERYTHING UP TO THIS POINT IS CORRECT ++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Define the number of predictors and the number of observations
m = len(X[0])
n = len(X)
y = y.reshape((n,1))

# Create a vector to store the initial predictions (which are all 0 to start).
mu_hats = np.zeros((n,1))
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


# DO I NEED TO LEAVE

# Calculate the gamma value
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
# At this point, we've added one variable to the model. Now to and number 2.
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

# Calculate the gamma value
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
# At this point, we've added one variable to the model. Now to and number 3.
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

# Calculate the gamma value
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
# At this point, we've added one variable to the model. Now to and number 4.
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

"""
# Calculate the gamma value
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
"""
gamma = bigC_hat / A_a


# Update our response predictions.
mu_hats += (gamma * u_a)


# Calculate the betas for the model associated with the new mu_hats.
new_betas = np.dot(npla.pinv(X), mu_hats).reshape(1,m)
models = np.concatenate((models, new_betas))

# Calculate and append the MSE of the current model.
mse_vals.append(((y - mu_hats)*(y- mu_hats)).sum())





















"""
# Initialize the predicted values all to be 0
preds = np.zeros((len(response),1))
# Intialize a DF to hold our successive models.
coefs = pd.DataFrame(np.zeros((1, len(df_cols))), columns = df_cols)
# Initialized the residuals to be the response values
resids = response - preds


# LAR step 2.8 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate the correlations of the predictors with the residuals
corrs = {}
for col in df_cols:
    current_col = np.array(x_mat[col]).reshape(len(x_mat), 1)
    corrs[col] = pearsonr(current_col, resids)[0][0]
# corrs = ((x_mat.T).dot((response - preds))).T

# Find the predictor with the highest correlation with the residual
varMaxCorr = max(corrs, key=lambda y: abs(corrs[y]))    


# Adjust the beta coefficient of varMaxCorr, recalculate resids, and recompute
# corrs. Do this until another variable has correlation magnitude equal to that
# of varMaxCorr.
# LAR step 2.9 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
bigC = abs(corrs[varMaxCorr])
active = [varMaxCorr]
# LAR step 2.10 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Determine the sign of the current variable under consideration.
s_j = np.sign(corrs[varMaxCorr])
# Initialize an empty X_a matrix
X_a = pd.DataFrame()
# Calculate the X_a matrix (add a column for each variable in the active set).
for var in active:
    X_a[var] = s_j * cement[var]
# Calculate the G_a matrix
G_a = np.dot(X_a.T, X_a)
# Designate the ones_a vector for the current active set.
ones_a = np.ones((len(active),1))
# Calculate the A_a matrix
A_a = 1/(np.sqrt((ones_a.T).dot(npla.inv(G_a)).dot(ones_a)[0][0]))
# Calculate the w_a vector
w_a = A_a * (npla.inv(G_a)).dot(ones_a)
# Calculate the equiangular vector u_a
u_a = X_a.dot(w_a)
# LAR step 2.11 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Compute a_vec
a_vec = ((X.T).dot(u_a)).T
# LAR step 2.12 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Determine the set of variables which is the complement of the active set.
active_compl = [var for var in list(x_mat) if var not in active]
# Calculate gamma_hat
gammas = []
for var in active_compl:
    gamma1 = (bigC - corrs[var])/(A_a - a_vec[var][0])
    gamma2 = (bigC + corrs[var])/(A_a + a_vec[var][0])
    if(gamma1 > 0.): gammas.append(gamma1)
    if(gamma2 > 0.): gammas.append(gamma2)
    print('var:', var)
    print('g1: ', gamma1)
    print('g2: ', gamma2)
gamma_hat = min(gammas)
# Update the preds vector
preds = preds + gamma_hat * u_a
# Update the betas
sa = X_a
sb = u_a * gamma_hat
sx = npla.lstsq(sa, sb)
betas = np.zeros((1, len(df_cols)))
for i, j in enumerate(active):
    betas[j] += sx[0][i] * sign[j]
"""





"""
# Following the naming and notational conventions of the LARS paper.

# Create the matrix of predictors (X), vector of response values (y), 
# the number of covariates (m), and the number of observations (n).
y = copy.deepcopy(response)
m = len(X[0])
n = len(X)

# Create a set object which will track the active set of predictors (predictors
# which have entered the model).
active = set()
# Initialize the vector of current predictions to 0 as given by the LAR paper.
mu_hats = np.zeros((n,1))
# Initialize the vector of residuals (initially it'll just be response values).
resids = y - mu_hats
# Calculate the correlations of each predictor with the residuals.
corrs = ((X.T).dot(resids))
j = np.argmax(np.abs(corrs[:,0]))
# Add the variable with highest correlation to the set of active predictors.
active.add(j)
# Initialize the current betas all to 0 (aligns with the logic of starting all
# predictions, mu_hats, at 0).
current_betas = np.zeros((m,1))
# Initialize a vector to store the signs for each variable (these signs are 
# either +1 or -1, depending on the sign of the correlation of that variable
# with the residuals during that LAR step).
signs = np.zeros((m,1))
# Set the sign according to which variable has the highest absolute correlation
# with the residuals (first variable that will be added to the model).
signs[j] = np.sign(corrs[j,0])

# Set up a NumPy array to track the values of the coefficients
all_betas = np.zeros((m,m))

# Create a list to store MSE values
mse_vals = []

# To complete the algorithm, we have to take the number of steps which there
# are number of variables.
for var in range(m):
    # Calculate the current residuals.
    resids = y - mu_hats
    # Calculate the MSE of the current model and add it to the mse_vals list.
    mse_vals.append((np.sqrt((resids * resids).sum())))
    # Generate predictions from our current set of betas.
    current_beta_preds = np.dot(X.T, resids)
    # Calculate correlations (with residuals) again. (We did nearly all of 
    # these same things above before the for loop).
    corrs = ((X.T).dot(resids))
    # Now we will calculate matrices, vectors, and constants which are given
    # in the LAR paper for computing how coefficients are moved equiangularly.
    # The X_a matrix includes only the columns of X whose variables are in the
    # active set. The values of each column are multiplied by their 
    # corresponding sign (direction of correlation with the residuals).
    X_a = X[ : , list(active)]
    for i, j in enumerate(active):
        X_a[:,i] *= signs[j]
    # The G_a matrix is just an |a|x|a| matrix (|a| = cardinality of active).
    G_a = np.dot(X_a.T, X_a)
    # Designate the ones_a vector for the current active set.
    ones_a = np.ones((len(active),1))
    # Calculate the A_a matrix
    A_a = 1/(np.sqrt((ones_a.T).dot(npla.inv(G_a)).dot(ones_a)[0][0]))
    # Calculate the w_a vector.
    w_a = A_a * (npla.inv(G_a)).dot(ones_a)
    # Calculate the equiangular vector u_a
    u_a = X_a.dot(w_a)
    # Compute a_vec (inner product vector of the X matrix and equiangular vec).
    # Pretty sure this is projecting onto the X matrix using our vector which
    # equiangularly extends the coefficients for the given active set.
    a_vec = np.dot(X.T, u_a)
    # Create a value for gamma_hat (this is used in the update step formula).
    gamma_hat = None
    # Find the largest correlation value.
    bigC = (np.abs(corrs)).max()
    # Next, we will be computing the gamma value for our update step. We have 
    # to do this by calculating candidate gammas for each covariate which isn't
    # in the active set (the active set complement). Since we've already found,
    # and added, the highest-correlated var to thea ctive set, we'll just do
    # m-1 loops for the remaining variables which can be added.
    if(var < m-1):
        # Initialize a place to store the index of the next variable to add, as
        # well as the sign of the next variable (will be either -1 or +1).
        j_next = None
        sign_next = 0
        # Loop over all variables, checking only those which are in the 
        # complement of the active set.
        for j in range(m):
            # If a variable is in the active set we skip over it.
            if(j in active): continue
            # Compute the two candidate gammas for each var in the complement
            # of the active set.
            gamma1 = ((bigC - corrs[j])/(A_a - a_vec[j][0]))[0]
            gamma2 = ((bigC + corrs[j])/(A_a + a_vec[j][0]))[0]
            # The LAR paper specifies that we can only consider positive gamma
            # values. We're searching for the lowest positive gamma_hat
            # candidate and the corresponding j_next and sign_next values.
            if((gamma1 > 0.) and ((gamma_hat is None) or (gamma1 < gamma_hat))):
                gamma_hat = gamma1
                j_next = j
                sign_next = np.sign(corrs[j,0])
            # Same checks for the other candidate gamma_hat
            if((gamma2 > 0.) and ((gamma_hat is None) or (gamma2 < gamma_hat))):
                gamma_hat = gamma2
                j_next = j
                sign_next = np.sign(corrs[j,0])
    # If we didn't find a gamma large enough, then this the new new gamma.
    else: gamma_hat = (bigC / A_a)
    # Next, we'll move on to calculating the new updated beta values using 
    # least square functionality from NumPy.
    ols_sol = npla.lstsq(X, (gamma_hat * u_a))
    for i, j in enumerate(active):
        current_betas[j] += ols_sol[0][i] * signs[j]
    # Now update our predictions (the mu_hats, normally the y_hats).
    mu_hats = np.dot(X, current_betas)
    # We'll also add the j_next we found by searching for gamma_hat above, as
    # well as updating the signs array.
    active.add(j_next)
    signs[j_next] = sign_next
    # We append the updated current_betas to the all_betas 2D array.
    all_betas[var, : ] = current_betas.reshape(m)
"""











# Create the coefficient progression plot.
custom_lines = [Line2D([0], [0], color='#a6611a', lw=4),
                Line2D([0], [0], color='#dfc27d', lw=4),
                Line2D([0], [0], color='#80cdc1', lw=4),
                Line2D([0], [0], color='#018571', lw=4)]

    
sum_coef = np.sum(np.abs(models), 1)          

ax = plt.gca()
plt.plot(sum_coef, models[:,0], color='#a6611a', lw=4)
plt.plot(sum_coef, models[:,1], color='#dfc27d', lw=4)   
plt.plot(sum_coef, models[:,2], color='#80cdc1', lw=4)        
plt.plot(sum_coef, models[:,3], color='#018571', lw=4)        

#plt.plot(sum_coef, models)
plt.title('From-Scratch: LASSO coefficients (via LARS)')
plt.ylabel('Standardized Coefficient Values')
ax.legend(custom_lines, ['aluminate',  'dicalcium', 'ferrite', 'silicate'])
plt.xlabel('Sum of the Absolute Value of the Coefficients')
plt.show()





















