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



from ClOp_calc import lm_analysis_ClOp
from prediction_ML_pipeline import data_preprocessing, prediction_feature, add_date_ticker, extract_info_from_filename, hid_outside_spread_tag
from order_imbalance import order_imbalance, combined_order_imbalance, conditional_order_imbalance, iceberg_order_imbalance



def process_and_train_xgb(archive_path, model_path, model_name, params,
                          num_boost_round=10, chunk_size=20000):
    """
    Extract 7z file, process CSVs in chunks, and train XGBoost classifier.
    """
    
    # Lists to accumulate validation predictions and true labels
    all_preds = []
    all_labels = []

    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        filenames = archive.getnames()
        orderbook_files = [f for f in filenames if 'orderbook' in f]
        message_files = [f for f in filenames if 'message' in f]
        # Process matching orderbook and message files for each trading day
        for orderbook_file, message_file in zip(orderbook_files, message_files):
            extracted_files = archive.read([orderbook_file, message_file])
            orderbook_stream = io.BytesIO(extracted_files[orderbook_file].read())
            message_stream = io.BytesIO(extracted_files[message_file].read())
            print("Processed files:", orderbook_file, message_file, flush=True)

            orderbook_chunk = pd.read_csv(orderbook_stream, header=None, usecols=[0, 1, 2, 3])
            message_chunk = pd.read_csv(message_stream, header=None, usecols=[0, 1, 2, 3, 4, 5])

            # Obtain date and ticker info from df
            ticker, date = extract_info_from_filename(message_file)

            # Clean dataframe and add ticker, date info
            message_chunk = add_date_ticker(message_chunk, date, ticker)

            # Preprocess dataframes
            message_chunk, orderbook_chunk = data_preprocessing(message_chunk, orderbook_chunk, ticker_name=ticker)

            # Obtain features for prediction using dfs
            X, y = prediction_feature(message_chunk, orderbook_chunk, labelled=True, standardise=True)

            # Map classes from -1 to 0
            y = y.replace(-1, 0)
            y = y.astype(int)

            # Split each chunk into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            
            dtrain_chunk = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dval_chunk = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

            evals = [(dtrain_chunk, 'train'), (dval_chunk, 'eval')]

            # Update model with new chunk for partial fitting
            if 'booster' in locals():
                booster = xgb.train(params, dtrain_chunk, num_boost_round, evals, xgb_model=booster)
            else:
                booster = xgb.train(params, dtrain_chunk, num_boost_round, evals)

            # Predict on the validation set for the current chunk
            preds = booster.predict(dval_chunk)
            preds = (preds > 0.5).astype(int)  # Thresholding for binary classification

            # Accumulate predictions and true labels
            all_preds.extend(preds)
            all_labels.extend(y_val)

    # Compute overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {overall_accuracy}", flush=True)

    # Save the final model to the specified folder
    model_path = os.path.join(model_path, model_name)
    booster.save_model(model_path)

    return booster, overall_accuracy


def order_imbalance_calc(archive_path, delta_lst, model=None,
                         model_path=None, model_name=None, order_type='combined',
                         specific_date=None, ticker=None):
    """
    Extract 7z file, process CSVs, predict using the trained model and create OI dataframes dict.
    """
    # Load the model from the JSON file
    if not model:
        model_path = os.path.join(model_path, model_name)
        model = xgb.Booster()
        model.load_model(model_path)

    df_dict = {key: [] for key in delta_lst}
    
    # Filter for specific date if specified
    if specific_date:
        year = specific_date[:4]
        archive_path = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{year}-01-01_{year}-12-31_10.7z"

    # Obtain messages and orderbook dataframe for each trading day
    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        filenames = archive.getnames()
        if not specific_date:
            orderbook_files = [f for f in filenames if 'orderbook' in f ]
            message_files = [f for f in filenames if 'message' in f]
    
        else:
            orderbook_files = [f for f in filenames if 'orderbook' in f and specific_date in f]
            message_files = [f for f in filenames if 'message' in f and specific_date in f]

        for orderbook_file, message_file in zip(orderbook_files, message_files):

            extracted_files = archive.read([orderbook_file, message_file])
            orderbook_stream = io.BytesIO(extracted_files[orderbook_file].read())
            message_stream = io.BytesIO(extracted_files[message_file].read())
            print("Processed files:", orderbook_file, message_file, flush=True)

            # Read the entire CSV files
            orderbook_df = pd.read_csv(orderbook_stream, header=None, usecols=[0, 1, 2, 3])
            message_df = pd.read_csv(message_stream, header=None, usecols=[0, 1, 2, 3, 4, 5])
            ticker, date = extract_info_from_filename(message_file)

            # Process data for prediction
            message_df = add_date_ticker(message_df, date, ticker)
            message_df, orderbook_df = data_preprocessing(message_df, orderbook_df, ticker_name=ticker)

            if message_df.empty:
                continue

            X = prediction_feature(message_df, orderbook_df, labelled=False, standardise=True)
    
            # Convert the data to DMatrix
            dmatrix = xgb.DMatrix(X, enable_categorical=True)

            # Predict using the loaded model
            pred_prob = model.predict(dmatrix)
            preds = (pred_prob > 0.5).astype(int)  # Thresholding for binary classification
            preds[preds == 0] = -1
            preds = preds.astype(int)

            y_pred_df = pd.DataFrame({'pred_dir': preds,
                                    'pred_prob': pred_prob})
            y_pred_df.index = X.index

            # Tag direction of hidden liquidity execution outside of bid-ask spread
            y_pred_df = hid_outside_spread_tag(X, y_pred_df)

            # Calculate OI for each delta in delta_lst
            for delta in delta_lst:
                if order_type == 'vis' or order_type == 'hid' or order_type == 'all':
                    df_merged = order_imbalance(message_df, y_pred_df, orderbook_df, delta=delta)
                    df_merged.rename(columns={'order_imbalance': f'order_imbalance_{order_type}'}, inplace=True)
                
                elif order_type == 'combined':
                    df_merged = combined_order_imbalance(message_df, y_pred_df, orderbook_df, delta=delta)

                elif order_type == 'comb_iceberg':
                    df_merged = iceberg_order_imbalance(message_df, y_pred_df, orderbook_df, delta=delta)

                elif order_type == 'agg' or order_type == 'size':
                    df_merged = conditional_order_imbalance(message_df, y_pred_df, orderbook_df, delta=delta, condition=order_type)
                
                df_dict[delta].append(df_merged)
                
    # Concatenate the DataFrames for each key
    for key in df_dict:
        if df_dict[key]:  # Check if the list associated with the key is not empty
            df_dict[key] = pd.concat(df_dict[key])
        else:
            df_dict[key] = pd.DataFrame()
    return df_dict



# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data_in_chunks(df, chunk_size=100):
    """Utility function to yield data in chunks."""
    for start in range(0, len(df), chunk_size):
        yield df[start:start + chunk_size]

def calculate_t_values_adj_R2(model, df, X_coefficients, output, chunk_size=100):
    """Calculate t-values for the coefficients in the model and adjusted R^2."""
    residual_sum_of_squares = 0
    XtX_sum = np.zeros((len(X_coefficients) + 1, len(X_coefficients) + 1))  # Include intercept

    # Calculate total sum of squares
    y_mean = np.mean(df[output])
    total_sum_of_squares = 0

    for chunk in get_data_in_chunks(df, chunk_size):
        # Compute RSS for current chunk
        X_chunk = chunk[X_coefficients].fillna(0).replace(-np.inf, 0).replace(np.inf, 0).values
        y_chunk = chunk[output].fillna(0).replace(-np.inf, 0).values
        y_chunk_pred = model.predict(X_chunk)
        residuals = y_chunk - y_chunk_pred
        residual_sum_of_squares += np.dot(residuals, residuals)
        
        # Calculate total sum of squares incrementally
        total_sum_of_squares += np.sum((y_chunk - y_mean) ** 2)

        # Add intercept to X matrix directly
        intercept = np.ones((X_chunk.shape[0], 1))
        X_chunk_with_intercept = np.hstack((intercept, X_chunk))
        XtX_sum += np.dot(X_chunk_with_intercept.T, X_chunk_with_intercept)

    # Calculate the variance of the residuals
    n = len(df)
    p = len(X_coefficients)
    degrees_of_freedom = n - p - 1
    variance_of_residuals = residual_sum_of_squares / degrees_of_freedom

    # Calculate the covariance matrix
    covariance_matrix = variance_of_residuals * inv(XtX_sum)
    standard_errors = np.sqrt(np.diag(covariance_matrix)[1:])

    # Calculate t-values
    t_values = model.coef_ / standard_errors

    # Calculate R-squared and adjusted R-squared
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / degrees_of_freedom)

    return t_values, adjusted_r_squared
    

def lm_analysis(df, order_type='combined', predictive=True, ret_type='log_ret',
                momentum=False):
    '''Compute linear regression coefficients using SGD for specified return and order type'''
    if ret_type == 'log_ret':
        output = "fut_log_ret" if predictive else "log_ret"
    
    elif ret_type == 'log_ret_ex':
        output = "fut_log_ret_ex" if predictive else "log_ret_ex"

    elif ret_type == 'weighted_mp':
        output = 'fut_weighted_log_ret' if predictive else "weighted_log_ret"

    elif ret_type == 'tClose':
        output = "fret_tClose" if predictive else "ret_tClose"
    
    elif ret_type == 'ClOp' or ret_type == 'ClCl' or ret_type == 'ClOp_ex' or ret_type == 'ClCl_ex' or ret_type == 'adjClOp':
        output = f"fret_{ret_type}"

    elif ret_type == 'daily_ret' or ret_type == 'daily_ret_ex':
        output = f"fut_{ret_type}"

    # Initialize the SGDRegressor
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-6)

    # Fit the model in chunks
    coefficients_dict = {
        'vis': ['order_imbalance_vis'],
        'hid': ['order_imbalance_hid'],
        'all': ['order_imbalance_all'],
        'combined': ['order_imbalance_vis', 'order_imbalance_hid'],
        'comb_iceberg': ['order_imbalance_vis', 'order_imbalance_hid', 'order_imbalance_ib'],
        'size': ['order_imbalance_vis', 'order_imbalance_small', 
                 'order_imbalance_medium', 'order_imbalance_large'],
        'agg': ['order_imbalance_vis', 'order_imbalance_agg_low',
                'order_imbalance_agg_mid', 'order_imbalance_agg_high']
    }

    X_coefficients = coefficients_dict[order_type]

    X_coefficients += ['SMB', 'HML', 'RF', 'CMA', 'RMW']


    if momentum and ret_type == 'weighted_mp':
        X_coefficients += ['weighted_log_ret']

    elif momentum and ret_type == "log_ret_ex":
        X_coefficients += ['log_ret_ex']
    
    elif momentum and ret_type == 'log_ret':
        X_coefficients += ['log_ret']

    elif momentum and ret_type == 'tClose':
        X_coefficients += ['ret_tClose']
    
    elif momentum and (ret_type == 'daily_ret' or ret_type == 'daily_ret_ex'):
        X_coefficients += [f'{ret_type}']
    
    elif momentum and ret_type == 'ClOp':
        X_coefficients += ['ClOp']

    # Using linear regression
    # X = df[X_coefficients].fillna(0).replace(-np.inf, 0).replace(np.inf, 0)
    # y = df[output].fillna(0).replace(-np.inf, 0).replace(np.inf, 0)

    # X = sm.add_constant(X)
    # model = sm.OLS(y, X).fit()

    # intercept = model.params[0]
    # coefficients = model.params[1:]

    # t_values = model.tvalues[1:]
    # adj_r2 = model.rsquared_adj


    # return intercept, coefficients.tolist(), t_values.tolist(), adj_r2

    # Use SGD to obtain coefficients
    for chunk in get_data_in_chunks(df, chunk_size=40):
        try:            
            X_chunk = chunk[X_coefficients].fillna(0).replace(-np.inf, 0).replace(np.inf, 0).values
            y_chunk = chunk[output].fillna(0).replace(-np.inf, 0).replace(np.inf, 0).values



            # Check for NaNs
            if np.isnan(X_chunk).any() or np.isnan(y_chunk).any():
                logging.warning("NaNs detected in chunk data.")
                continue
            
            sgd_reg.partial_fit(X_chunk, y_chunk)
        except Exception as e:
            logging.error(f"Error processing chunk: {e}")
            continue

    try:
        logging.info("Model fit completed")

        coefficients = sgd_reg.coef_
        intercept = sgd_reg.intercept_[0]
        t_values, adj_r2 = calculate_t_values_adj_R2(sgd_reg, df, X_coefficients, output, chunk_size=100)

        logging.info("Coefficients and t_values obtained")
        return intercept, coefficients.tolist(), t_values.tolist(), adj_r2
    except Exception as e:
        logging.error(f"Error in final model fit: {e}")
        return [], []

def OI_results(df_dict, order_type='combined', predictive=True, ret_type='log_ret',
                momentum=False):
    # Build dataframe for regression using coefficients obtained from lm_analysis
    lm_results = []

    col_names_dict = {
        'vis': ['timeframe', 'intercept', 'adj_R2', 'params_vis', 'tvals_vis'],
        'hid': ['timeframe', 'intercept', 'adj_R2', 'params_hid', 'tvals_hid'],
        'all': ['timeframe', 'intercept', 'adj_R2', 'params_all', 'tvals_all'],
        'combined': ['timeframe', 'intercept', 'adj_R2', 'params_vis', 'tvals_vis', 'params_hid', 'tvals_hid'],
        'comb_iceberg': ['timeframe', 'intercept', 'adj_R2', 'params_vis', 'tvals_vis', 'params_hid',
                         'tvals_hid', 'params_ib', 'tvals_ib'],
        'agg': ['timeframe', 'intercept', 'adj_R2', 'params_vis', 'tvals_vis',
                'params_low', 'tvals_low', 'params_mid', 'tvals_mid', 'params_high', 'tvals_high'],
        'size': ['timeframe', 'intercept', 'adj_R2', 'params_vis', 'tvals_vis',
                 'params_small', 'tvals_small', 'params_mid', 'tvals_mid', 'params_large', 'tvals_large']

    }
    logging.info("Process started")
    logging.debug(f"DataFrames in df_dict: {list(df_dict.keys())}")
    
    # Compute linear regression type depending on daily or intraday delta
    for delta in df_dict:
        logging.info(f'Currently fitting for delta: {delta}')
        row_result = [delta]
        try:
            if delta != 'daily':
                intercept, coefficients, t_values, adj_r2 = lm_analysis(df_dict[delta], order_type=order_type, 
                                                    predictive=predictive, ret_type=ret_type, momentum=momentum)
            
            else:
                intercept, coefficients, t_values, adj_r2 = lm_analysis_ClOp(df_dict[delta], order_type=order_type)

            row_result += [intercept]
            row_result += [adj_r2]

            for coef, t_val in zip(coefficients, t_values):
                row_result += [coef]
                row_result += [t_val]

            lm_results.append(row_result)
        except Exception as e:
            logging.error(f"Error in lm_analysis for delta {delta}: {e}")
            continue
    
    logging.info("Process completed")
    logging.debug(f"LM Results: {lm_results}")

    # Build dataframe
    col_names = col_names_dict[order_type]
    col_names += ['params_SMB', 'tvals_SMB',
                    'params_HML', 'tvals_SMB',
                    'params_RF', 'tvals_RF',
                    'params_CMA', 'tvals_CMA',
                    'params_RMW', 'tvals_RMW']
    if momentum:
        col_names += ['params_momentum', 'tvals_momentum']


    return pd.DataFrame(lm_results, columns=col_names)
