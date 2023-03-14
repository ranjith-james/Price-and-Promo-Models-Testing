# -*- coding: utf-8 -*-
"""
@author: Ranjith James
"""

import scipy
import datetime as dt
import numpy as np
try:
    np.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass
import pymc as pm
import pandas as pd
import theano
import numpy as np
from scipy.optimize import minimize



pd.set_option('display.max_colwidth', None)

X=pd.read_csv(r"C:\Users\Ranjith James\Documents\PYMC_TRIAL\X.csv")
y=pd.read_csv(r"C:\Users\Ranjith James\Documents\PYMC_TRIAL\Y.csv")
y=np.array(y["0"])
df=pd.read_csv(r"C:\Users\Ranjith James\Documents\PYMC_TRIAL\df.csv")


### DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills
### Naive Priors Model - Basically will provide OLS results - used to test working
#### 
basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters

    beta_1 =pm.Normal('beta_1', mu=0, sigma=20)
    beta_2 = pm.Normal('beta_2', mu=0, sigma=20)
    beta_3 = pm.Normal('beta_3', mu=0, sigma=20)
    beta_4 = pm.Normal('beta_4', mu=0, sigma=20)
    beta_5 = pm.Normal('beta_5', mu=0, sigma=20)
    beta_6 = pm.Normal('beta_6', mu=0, sigma=20)
    alpha =  pm.Normal('alpha', mu=0, sigma=4)
    sigma =  pm.HalfCauchy('sigma', beta=10)

    # Expected value of outcome
    mu = alpha + beta_1*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_ln_acv'] + beta_2*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_ln_base_price'] + beta_3*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_yearly_seasonality_segment']\
    + beta_4*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_christmas_flg'] +  beta_5*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_new_year_flg']\
    + beta_6*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_time_trend']
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)




with basic_model:
    # draw 1000 posterior samples
    idata = pm.sample()
    

import arviz as az
az.plot_trace(idata)


def objective2(params):
    alpha,beta_1,beta_2,beta_3,beta_4,beta_5,beta_6= params
    y_pred = alpha + beta_1*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_ln_acv'] + beta_2*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_ln_base_price'] + beta_3*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_yearly_seasonality_segment']\
    + beta_4*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_christmas_flg'] +  beta_5*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_new_year_flg']\
    + beta_6*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_time_trend']
    
    return np.sum((y_pred - y)**2)

#params0 = [7.029238,0.843515,-0.30,1.118830,0.080261,0.028600,0.087582,-0.840806,0.048184,0.035065]
bounds = [(0, None),(0.1, 2),(-3.5, -0.1),(None, None), (None, None),(None, None),(None, None)]

params0 = [1,1,-1,1,1,1,1]

# Minimize the function subject to the bounds
result = minimize(objective2,params0, bounds=bounds,method='trust-constr')

print(result)
result.x

##########################################################
### Informed Priors based on OLS results
#######################################################


X=pd.read_csv(r"C:\Users\Ranjith James\Documents\PYMC_TRIAL\X_P.csv")
y=pd.read_csv(r"C:\Users\Ranjith James\Documents\PYMC_TRIAL\Y_P.csv")
y=np.array(y["0"])
df=pd.read_csv(r"C:\Users\Ranjith James\Documents\PYMC_TRIAL\df_P.csv")



basic_model = pm.Model()

with basic_model:
    ### Setting priors based on Distributions from all ols models for the same country category
    # Priors for unknown model parameters
    beta_1_mean = df[df['Variable'].str.contains("ln_acv")]['Prior'].mean()
    beta_1_sd = df[df['Variable'].str.contains("ln_acv")]['Prior'].std()
    beta_1 =pm.TruncatedNormal('beta_1', mu=beta_1_mean, sigma=beta_1_sd,upper=2,lower=0.01)

    beta_2_mean = df[df['Variable'].str.contains("base_price")]['Prior'].mean()
    beta_2_sd = df[df['Variable'].str.contains("base_price")]['Prior'].std()
    beta_2 = pm.TruncatedNormal('beta_2', mu=beta_2_mean, sigma=beta_2_sd,upper=-0.1,lower=-3.5)
    #beta_2 = pm.Normal('beta_2', mu=beta_2_mean, sigma=beta_2_sd)
    
    beta_3_mean = df[df['Variable'].str.contains("yearly_seasonality_segment")]['Prior'].mean()
    beta_3_sd = df[df['Variable'].str.contains("yearly_seasonality_segment")]['Prior'].std()
    beta_3 = pm.Normal('beta_3', mu=beta_3_mean, sigma=beta_3_sd)

    beta_4_mean = df[df['Variable'].str.contains("christmas_flg")]['Prior'].mean()
    beta_4_sd = df[df['Variable'].str.contains("christmas_flg")]['Prior'].std()
    beta_4 =  pm.Normal('beta_4', mu=beta_4_mean, sigma=beta_4_sd)

    beta_5_mean = df[df['Variable'].str.contains("new_year_flg")]['Prior'].mean()
    beta_5_sd = df[df['Variable'].str.contains("new_year_flg")]['Prior'].std()
    beta_5 =  pm.Normal('beta_5', mu=beta_5_mean, sigma=beta_5_sd)

    beta_6_mean = df[df['Variable'].str.contains("hills_time_trend")]['Prior'].mean()
    beta_6_sd = df[df['Variable'].str.contains("hills_time_trend")]['Prior'].std()
    beta_6 =  pm.Normal('beta_6', mu=beta_6_mean, sigma=beta_6_sd)
    
    beta_7_mean = df[df['Variable'].str.contains("ln_bp_ratio")]['Prior'].mean()
    beta_7_sd = df[df['Variable'].str.contains("ln_bp_ratio")]['Prior'].std()
    beta_7 =  pm.TruncatedNormal('beta_7', mu=beta_7_mean, sigma=beta_7_sd,upper=-0.01,lower=-1.25)
    
    beta_8_mean = df[df['Variable'].str.contains("hills_blackfriday_flg")]['Prior'].mean()
    beta_8_sd = df[df['Variable'].str.contains("hills_blackfriday_flg")]['Prior'].std()
    beta_8 =  pm.Normal('beta_8', mu=beta_8_mean, sigma=beta_8_sd)

    beta_9_mean = df[df['Variable'].str.contains("december_flg")]['Prior'].mean()
    beta_9_sd = df[df['Variable'].str.contains("december_flg")]['Prior'].std()
    beta_9 =  pm.Normal('beta_9', mu=beta_9_mean, sigma=beta_9_sd)

    alpha =  pm.Normal('alpha', mu=0, sigma=4)

#    sigma =  pm.HalfNormal('sigma', sigma=1)
    sigma =  pm.HalfCauchy('sigma', beta=10)

    # Expected value of outcome
    mu = alpha + beta_1*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_ln_acv'] + beta_2*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_ln_base_price'] + beta_3*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_yearly_seasonality_segment']\
    + beta_4*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_christmas_flg'] +  beta_5*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_new_year_flg']\
    + beta_6*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_time_trend'] + beta_7*X['DOG FOOD_GENERAL MILLS_BLUE_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_REM SEGMENT_10-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_ln_bp_ratio']\
        +beta_8*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_blackfriday_flg'] + beta_9*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_december_flg']
    # Likelihood (sampling distribution) of observations
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

with basic_model:
    # draw 1000 posterior samples
    idata = pm.sample()
    

az.plot_trace(idata)
res=az.summary(idata, round_to=10)
res.to_csv(r"C:\Users\Ranjith James\Documents\PYMC_TRIAL\bay_res_acc_2.csv")


##### Contraint Optimized Regression using Scipy - Trust Region Optimizer 


# Define the function to minimize
def objective(params):
    alpha,beta_1,beta_2,beta_3,beta_4,beta_5,beta_6,beta_7,beta_8,beta_9 = params
    y_pred = alpha + beta_1*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_ln_acv'] + beta_2*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_ln_base_price'] + beta_3*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_yearly_seasonality_segment']\
    + beta_4*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_christmas_flg'] +  beta_5*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_new_year_flg']\
    + beta_6*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_time_trend'] + beta_7*X['DOG FOOD_GENERAL MILLS_BLUE_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_REM SEGMENT_10-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_ln_bp_ratio']\
        +beta_8*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_blackfriday_flg'] + beta_9*X['DOG FOOD_HILLS_SCIENCE DIET_DRY_SINGLE PACK_8.1-20.9LB (DD-MD)_EVERYDAY_6-petco total b&m ta-dog food-dry-usa-pet_care_petco_bm-hills_december_flg']
    
    return np.sum((y_pred - y)**2)

### OLS Quantiles , BASE price 

# Define the upper and lower bounds for the coefficients
bounds = [(0, None),(0.1, 2),(-3.5, -0.1),(None, None), (None, None),(None, None),(None, None),(-1.25, -0.01),(None, None),(None, None)]

## Initializing Priors from OLS results 
#params0 = [7.029238,0.843515,-0.30,1.118830,0.080261,0.028600,0.087582,-0.840806,0.048184,0.035065]

params0 = [1,1,-1,1,1,1,1,-1,1,1]

# Minimize the function subject to the bounds
result = minimize(objective,params0, bounds=bounds,method='trust-constr')

# Print the results
print(result)
result.x

