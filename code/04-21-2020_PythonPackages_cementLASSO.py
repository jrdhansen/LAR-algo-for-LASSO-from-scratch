'''
File name      : PythonPackages_cementLASSO.py
Author         : Jared Hansen
Python version : 3.7.3

DESCRIPTION:
Using pre-made Python packages to fit a LASSO model, create a coefficient
progression plot, and determine the true MSE of the best model (where best is
the model with the lowest MSE).

https://www.kirenz.com/post/2019-08-12-python-lasso-regression-auto/
'''


#++++ IMPORT STATEMENTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLarsCV
from matplotlib.lines import Line2D


#++++ PROCEDURAL CODE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Read in the data
data_path = 'C:/__JARED/__USU_Sp2020/stat6100_AdvRegression/finalProject/data'
cement = pd.read_csv(data_path+'/haldCement.csv')

# Split the data into a matrix of predictors and a vector of response values.
x_mat = cement[['aluminate', 'silicate', 'ferrite', 'dicalcium']]
y_vec = cement['heat']

# Create the LASSO model using LOOCV (PRESS).
model = LassoLarsCV(fit_intercept = True, normalize=True, cv=12,
                    precompute=False).fit(x_mat, y_vec)
# Take a look at the model coefficients for the best model.
print(dict(zip(x_mat.columns, model.coef_)))
model.intercept_
#best_betas = [72.2200604, 1.42376893, 0.40740646, 0., -0.2346226 ]

# Create the coefficient progression plot.
custom_lines = [Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='orange', lw=4),
                Line2D([0], [0], color='green', lw=4)]
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.ylabel('Regression Coefficients')
ax.legend(custom_lines, ['dicalcium', 'aluminate', 'silicate', 'ferrite'])
plt.xlabel('-log(alpha)')
plt.title('Hald cement: Python LassoLarsCV Coefficient Progression')
plt.show()
