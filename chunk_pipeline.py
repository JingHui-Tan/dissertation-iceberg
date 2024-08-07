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


from prediction_ML_pipeline import data_preprocessing, prediction_feature, add_date_ticker, extract_info_from_filename, hid_outside_spread_tag
from order_imbalance import order_imbalance, combined_order_imbalance, conditional_order_imbalance, iceberg_order_imbalance



# def process_and_train_xgb(archive_path, model_path, params, num_boost_round=10, chunk_size=20000):
#     """
#     Extract 7z file, process CSVs in chunks, and train HistGradientBoostingClassifier.
#     """
    
#     # Lists to accumulate validation predictions and true labels
#     all_preds = []
#     all_labels = []

#     with py7zr.SevenZipFile(archive_path, mode='r') as archive:
#         filenames = archive.getnames()
#         orderbook_files = [f for f in filenames if 'orderbook' in f]
#         message_files = [f for f in filenames if 'message' in f]
#         # Process matching orderbook and message files
#         for orderbook_file, message_file in zip(orderbook_files, message_files):
#             extracted_files = archive.read([orderbook_file, message_file])
#             orderbook_stream = io.BytesIO(extracted_files[orderbook_file].read())
#             message_stream = io.BytesIO(extracted_files[message_file].read())
#             print("Processed files:", orderbook_file, message_file)

#             orderbook_iter = pd.read_csv(orderbook_stream, chunksize=chunk_size, usecols=[0, 1, 2, 3])
#             message_iter = pd.read_csv(message_stream, chunksize=chunk_size, usecols=[0, 1, 2, 3, 4, 5])
#             ticker, date = extract_info_from_filename(message_file)

#             for orderbook_chunk, message_chunk in zip(orderbook_iter, message_iter):
#                 # Merge the two chunks on a common key, adjust key names if necessary
#                 message_chunk = add_date_ticker(message_chunk, date, ticker)
#                 message_chunk, orderbook_chunk = data_preprocessing(message_chunk, orderbook_chunk, ticker_name=ticker)
#                 X, y = prediction_feature(message_chunk, orderbook_chunk, labelled=True, standardise=True)
#                 # Map classes from -1 to 0
#                 y = y.replace(-1, 0)
#                 y = y.astype(int)


#                 # Split each chunk into training and validation sets
#                 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
                
#                 dtrain_chunk = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
#                 dval_chunk = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

#                 evals = [(dtrain_chunk, 'train'), (dval_chunk, 'eval')]

#                 # Update model with new chunk
#                 if 'booster' in locals():
#                     booster = xgb.train(params, dtrain_chunk, num_boost_round, evals, xgb_model=booster)
#                 else:
#                     booster = xgb.train(params, dtrain_chunk, num_boost_round, evals)
    
#                 # Predict on the validation set for the current chunk
#                 preds = booster.predict(dval_chunk)
#                 preds = (preds > 0.5).astype(int)  # Thresholding for binary classification

#                 # Accumulate predictions and true labels
#                 all_preds.extend(preds)
#                 all_labels.extend(y_val)

#     # Compute overall accuracy
#     overall_accuracy = accuracy_score(all_labels, all_preds)
#     print(f"Overall Accuracy: {overall_accuracy}")

#     # print(f"Overall Accuracy: {test_correct / test_count}")
#     # Save the final model to the specified folder
#     model_path = os.path.join(model_path, 'xgboost_model.json')
#     booster.save_model(model_path)

#     return booster


def process_and_train_xgb(archive_path, model_path, params, num_boost_round=10, chunk_size=20000):
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
        # Process matching orderbook and message files
        for orderbook_file, message_file in zip(orderbook_files, message_files):
            extracted_files = archive.read([orderbook_file, message_file])
            orderbook_stream = io.BytesIO(extracted_files[orderbook_file].read())
            message_stream = io.BytesIO(extracted_files[message_file].read())
            print("Processed files:", orderbook_file, message_file)

            orderbook_chunk = pd.read_csv(orderbook_stream, header=None, usecols=[0, 1, 2, 3])
            message_chunk = pd.read_csv(message_stream, header=None, usecols=[0, 1, 2, 3, 4, 5])
            ticker, date = extract_info_from_filename(message_file)

            message_chunk = add_date_ticker(message_chunk, date, ticker)

            message_chunk, orderbook_chunk = data_preprocessing(message_chunk, orderbook_chunk, ticker_name=ticker)
            X, y = prediction_feature(message_chunk, orderbook_chunk, labelled=True, standardise=True)
            # Map classes from -1 to 0
            y = y.replace(-1, 0)
            y = y.astype(int)

            # Split each chunk into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            
            dtrain_chunk = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dval_chunk = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

            evals = [(dtrain_chunk, 'train'), (dval_chunk, 'eval')]

            # Update model with new chunk
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
    print(f"Overall Accuracy: {overall_accuracy}")

    # Save the final model to the specified folder
    model_path = os.path.join(model_path, 'xgboost_model.json')
    booster.save_model(model_path)

    return booster



## Usage
# archive_path = 'yourfile.7z'
# model = process_and_train_hist_gb(archive_path)

def order_imbalance_calc(archive_path, model,
                         delta_lst, order_type='combined'):
    """
    Extract 7z file, process CSVs, predict using the trained model and create OI dataframes dict.
    """
    # Load the trained model
    df_dict = {key: [] for key in delta_lst}
    

    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        filenames = archive.getnames()
        orderbook_files = [f for f in filenames if 'orderbook' in f]
        message_files = [f for f in filenames if 'message' in f]

        for orderbook_file, message_file in zip(orderbook_files, message_files):
            extracted_files = archive.read([orderbook_file, message_file])
            orderbook_stream = io.BytesIO(extracted_files[orderbook_file].read())
            message_stream = io.BytesIO(extracted_files[message_file].read())
            print("Processed files:", orderbook_file, message_file)

            # Read the entire CSV files
            orderbook_df = pd.read_csv(orderbook_stream, header=None, usecols=[0, 1, 2, 3])
            message_df = pd.read_csv(message_stream, header=None, usecols=[0, 1, 2, 3, 4, 5])

            ticker, date = extract_info_from_filename(message_file)

            # Process data for prediction
            message_df = add_date_ticker(message_df, date, ticker)
            message_df, orderbook_df = data_preprocessing(message_df, orderbook_df, ticker_name=ticker)
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


            for delta in delta_lst:
                if order_type == 'vis' or order_type == 'hid' or order_type == 'combined':
                    df_merged = combined_order_imbalance(message_df, y_pred_df, orderbook_df, delta=delta)

                elif order_type == 'comb_iceberg':
                    df_merged = iceberg_order_imbalance(message_df, y_pred_df, orderbook_df, delta=delta)

                elif order_type == 'agg' or order_type == 'size':
                    df_merged = conditional_order_imbalance(message_df, y_pred_df, orderbook_df, delta=delta, condition=order_type)
            
                df_dict[delta].append(df_merged)

    # Concatenate the DataFrames for each key
    for key in df_dict:
        df_dict[key] = pd.concat(df_dict[key])

    return df_dict

# Function to yield chunks of data
def get_data_in_chunks(df, chunk_size=100):
    for start in range(0, df.shape[0], chunk_size):
        end = min(start + chunk_size, df.shape[0])
        yield df.iloc[start:end]


def calculate_t_values(model, X, y):
    # Assuming we fit the model on the entire dataset to calculate residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    residual_sum_of_squares = np.sum(residuals**2)

    # Calculate the variance of the residuals
    degrees_of_freedom = X.shape[0] - X.shape[1] - 1
    variance_of_residuals = residual_sum_of_squares / degrees_of_freedom

    # Calculate the standard errors of the coefficients
    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
    covariance_matrix = variance_of_residuals * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))
    standard_errors = np.sqrt(np.diag(covariance_matrix)[1:])

    # Calculate t-values
    t_values = model.coef_ / standard_errors

    return t_values


def lm_analysis(df, order_type='combined', predictive=True, weighted_mp=False,
                momentum=False):
    
    if weighted_mp==False:
        output = "fut_log_ret" if predictive else "log_ret"
    else:
        output = 'fut_weighted_log_ret' if predictive else "weighted_log_ret"    

    # Initialize the SGDRegressor
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3)

    # Fit the model in chunks
    coefficients_dict = {
        'vis': ['order_imbalance_vis'],
        'hid': ['order_imbalance_hid'],
        'combined': ['order_imbalance_vis', 'order_imbalance_hid'],
        'comb_iceberg': ['order_imbalance_vis', 'order_imbalance_hid', 'order_imbalance_ib'],
        'size': ['order_imbalance_vis', 'order_imbalance_small', 
                 'order_imbalance_medium', 'order_imbalance_large'],
        'agg': ['order_imbalance_vis', 'order_imbalance_agg_low',
                'order_imbalance_agg_mid', 'order_imbalance_agg_high']
    }

    X_coefficients = coefficients_dict[order_type]

    if momentum and weighted_mp:
        X_coefficients += ['weighted_log_ret']
    
    elif momentum and not weighted_mp:
        X_coefficients += ['log_ret']

    for chunk in get_data_in_chunks(df, chunk_size=100):
        X_chunk = chunk[X_coefficients].values
        y_chunk = chunk[output].values
        sgd_reg.partial_fit(X_chunk, y_chunk)

    X = df[X_coefficients]
    y = df[output]
    print("Model fit completed")
    coefficients = sgd_reg.coef_
    t_values = calculate_t_values(sgd_reg, X, y)

    return coefficients.tolist(), t_values.tolist()
    

def OI_results(df_dict, order_type='combined', predictive=True, weighted_mp=False,
                momentum=False):
    lm_results = []

    col_names_dict = {
        'vis': ['timeframe', 'params_vis', 'tvals_vis'],
        'hid': ['timeframe', 'params_hid', 'tvals_hid'],
        'combined': ['timeframe', 'params_vis', 'tvals_vis', 'params_hid', 'tvals_hid'],
        'comb_iceberg': ['timeframe', 'params_vis', 'tvals_vis', 'params_hid',
                         'tvals_hid', 'params_ib', 'tvals_ib'],
        'agg': ['timeframe', 'params_vis', 'tvals_vis', 'params_hid', 'tvals_hid',
                'params_low', 'tvals_low', 'params_mid', 'tvals_mid', 'params_high', 'tvals_high'],
        'size': ['timeframe', 'params_vis', 'tvals_vis', 'params_hid', 'tvals_hid',
                 'params_small', 'tvals_small', 'params_mid', 'tvals_mid', 'params_large', 'tvals_large']

    }

    for delta in df_dict:
        row_result = [delta]
        # Need to be in the form of lists
        coefficients, t_values = lm_analysis(df_dict[delta], order_type=order_type, 
                                             predictive=predictive, weighted_mp=weighted_mp, momentum=momentum)

        for coef, t_val in zip(coefficients, t_values):
            row_result += [coef]
            row_result += [t_val]
        
        lm_results.append(row_result)
    
    return pd.DataFrame(lm_results, columns=col_names_dict[order_type])




    


                

