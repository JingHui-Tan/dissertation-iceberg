from chunk_pipeline import *
from prediction_ML_pipeline import save_dataframe_to_folder
import time
import os


def main():
    logging.getLogger().setLevel(logging.WARNING)

        ## ---------ADJUST THIS---------------

    ticker_lst = ['AES', 'ALB', 'AOS', 'APA', 'BEN', 'BXP', 'CPB',
                  'DVA', 'FFIV', 'FRT', 'HII', 'HRL', 'HRL', 'HSIC', 'INCY',
                  'MHK', 'NWSA', 'PNW', 'RL', 'TAP', 'WYNN']

    testing_year = "2019"
    params_year = "2018"
    order_type = 'combined'
    predictive = True
    ret_type = "log_ret_ex"
    momentum = False
    params_file_name = "all_log_ret_ex_2018.csv"

    strat_delta_lst = ['1min', '2min', '10min']

    ## ------------------------------------

    # Check if files exist
    params_folder_path = f"/nfs/home/jingt/dissertation-iceberg/data/output_results/{params_year}"
    model_path = "/nfs/home/jingt/dissertation-iceberg/data/output_folder"

    params_file_path = os.path.join(params_folder_path, params_file_name)
    params_df = pd.read_csv(params_file_path)

    # Check all tickers are in params_df and files exist
    for ticker in ticker_lst:
        file_path_test = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{testing_year}-01-01_{testing_year}-12-31_10.7z"

        if not os.path.exists(file_path_test):
            print(f"The file {file_path_test} does not exist.")
        
        if ticker not in params_df['ticker']:
            print(f"{ticker} not in params dataframe!")
        