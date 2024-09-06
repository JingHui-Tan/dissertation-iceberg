import py7zr
import pandas as pd
import io
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import statsmodels.formula.api as smf
from sklearn.linear_model import SGDRegressor
import os
import logging
from scipy.linalg import inv
import gc
import statsmodels.api as sm



from prediction_ML_pipeline import data_preprocessing, prediction_feature, add_date_ticker, extract_info_from_filename, hid_outside_spread_tag
from order_imbalance import order_imbalance, combined_order_imbalance, conditional_order_imbalance, iceberg_order_imbalance



def lm_analysis_ClOp(df, order_type='combined'):
    output = 'fret_ClOp'

    coefficients_dict = {
        'all': ['order_imbalance_all'],
        'combined': ['order_imbalance_vis', 'order_imbalance_hid'],
        'comb_iceberg': ['order_imbalance_vis', 'order_imbalance_hid', 'order_imbalance_ib']
    }

    X_coefficients = coefficients_dict[order_type]
    
    X_coefficients += ['ClOp']
    X_coefficients += ['SMB', 'HML', 'RF', 'CMA', 'RMW']

    X = df[X_coefficients]
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    X = sm.add_constant(X)
    y = df[output]

    model = sm.OLS(y, X)
    results = model.fit()
    pd.set_option("display.max_columns", None)
    print(X)
    print(y)

    intercept = results.params[0]
    params = results.params[1:].tolist()
    tvalues = results.tvalues[1:].tolist()
    adjusted_r_squared = results.rsquared_adj

    print(f"adj_r2: {adjusted_r_squared}")

    return intercept, params, tvalues, adjusted_r_squared



