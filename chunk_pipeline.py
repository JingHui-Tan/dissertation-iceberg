import py7zr
import pandas as pd
import io
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import statsmodels.formula.api as smf
from sklearn.linear_model import SGDRegressor


from prediction_ML_pipeline import data_preprocessing, prediction_feature, add_date_ticker, extract_info_from_filename
from order_imbalance import order_imbalance, combined_order_imbalance, conditional_order_imbalance, iceberg_order_imbalance



def process_and_train_hist_gb(archive_path, model_path, chunk_size=10000):
    """
    Extract 7z file, process CSVs in chunks, and train HistGradientBoostingClassifier.
    """

    model = HistGradientBoostingClassifier()
    test_count = 0
    test_correct = 0

    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        filenames = archive.getnames()
        orderbook_files = [f for f in filenames if 'orderbook' in f]
        message_files = [f for f in filenames if 'messages' in f]

        for orderbook_file, message_file in zip(orderbook_files, message_files):
            print((orderbook_file, message_file))
            with archive.read([orderbook_file, message_file]) as extracted_files:
                orderbook_stream = io.BytesIO(extracted_files[orderbook_file].read())
                message_stream = io.BytesIO(extracted_files[message_file].read())

                orderbook_iter = pd.read_csv(orderbook_stream, chunksize=chunk_size, usecols=[0, 1, 2, 3])
                message_iter = pd.read_csv(message_stream, chunksize=chunk_size, usecols=[0, 1, 2, 3, 4, 5])

                ticker, date = extract_info_from_filename(message_stream)

                for orderbook_chunk, message_chunk in zip(orderbook_iter, message_iter):
                    # Merge the two chunks on a common key, adjust key names if necessary
                    message_chunk = add_date_ticker(message_chunk, date, ticker)
                    message_chunk_mh, orderbook_chunk_mh = data_preprocessing(message_chunk, orderbook_chunk, ticker_name=ticker)
                    X, y = prediction_feature(message_chunk_mh, orderbook_chunk_mh, labelled=True, standardise=True)

                    # Split each chunk into training and validation sets
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

                    # Fit the model incrementally
                    model.partial_fit(X_train, y_train, classes=np.unique(y))

                    # Optional: Evaluate the model on the validation set of the current chunk
                    y_pred = model.predict(X_val)
                    chunk_accuracy = accuracy_score(y_val, y_pred)
                    test_correct += sum(y_pred == y_val)
                    test_count += len(y_val)
                    print(f'Chunk Accuracy: {chunk_accuracy}')

    print(f"Overall Accuracy: {test_correct / test_count}")
    joblib.dump(model, model_path)
    return model


## Usage
# archive_path = 'yourfile.7z'
# model = process_and_train_hist_gb(archive_path)

def order_imbalance_calc(archive_path, model_path,
                         delta_lst, order_type='combined'):
    """
    Extract 7z file, process CSVs, predict using the trained model and create OI dataframes dict.
    """
    # Load the trained model
    model = joblib.load(model_path)
    df_dict = {key: [] for key in delta_lst}
    

    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        filenames = archive.getnames()
        orderbook_files = [f for f in filenames if 'orderbook' in f]
        message_files = [f for f in filenames if 'messages' in f]

        for orderbook_file, message_file in zip(orderbook_files, message_files):
            with archive.read([orderbook_file, message_file]) as extracted_files:
                orderbook_stream = io.BytesIO(extracted_files[orderbook_file].read())
                message_stream = io.BytesIO(extracted_files[message_file].read())

                # Read the entire CSV files
                orderbook_df = pd.read_csv(orderbook_stream, header=None, usecols=[0, 1, 2, 3])
                message_df = pd.read_csv(message_stream, header=None, usecols=[0, 1, 2, 3, 4, 5])

                ticker, date = extract_info_from_filename(message_stream)

                # Process data for prediction
                message_df = add_date_ticker(message_df, date, ticker)
                message_df_mh, orderbook_df_mh = data_preprocessing(message_df, orderbook_df, ticker_name=ticker)
                X = prediction_feature(message_df_mh, orderbook_df_mh, labelled=False, standardise=True)

                # Predict using the trained model
                predictions = model.predict(X)

                for delta in delta_lst:
                    if order_type == 'vis' or order_type == 'hid' or order_type == 'combined':
                        df_merged = combined_order_imbalance(message_df_mh, predictions, orderbook_df_mh, delta=delta)

                    elif order_type == 'comb_iceberg':
                        df_merged = iceberg_order_imbalance(message_df_mh, predictions, orderbook_df_mh, delta=delta)

                    elif order_type == 'agg' or order_type == 'size':
                        df_merged = conditional_order_imbalance(message_df_mh, predictions, orderbook_df_mh, delta=delta, condition=order_type)
                
                    df_dict[delta].append(df_merged)

    # Concatenate the DataFrames for each key
    for key in df_dict:
        df_dict[key] = pd.concat(df_dict[key], ignore_index=True)

    return df_dict

# Function to yield chunks of data
def get_data_in_chunks(df, chunk_size=100):
    for start in range(0, df.shape[0], chunk_size):
        end = min(start + chunk_size, df.shape[0])
        yield df.iloc[start:end]


def calculate_t_values(X, y):
    pass

def lm_analysis(df, order_type='combined', predictive=True, weighted_mp=False,
                momentum=False):
    
    params_lst = []
    tvalues_lst = []
    
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

    coefficients = sgd_reg.coef_
    t_values = calculate_t_values(X, y)

    return coefficients, t_values
    

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
        coefficients, t_values = lm_analysis(df_dict[delta], order_type='combined', predictive=True, weighted_mp=False, momentum=False)

        for coef, t_val in zip(coefficients, t_values):
            row_result += [coef]
            row_result += [t_val]
        
        lm_results.append(row_result)
    
    return pd.DataFrame(lm_results, columns=col_names_dict[order_type])




    


                

