from chunk_pipeline import *
from prediction_ML_pipeline import save_dataframe_to_folder
import time
import os
from trading_strategy import combined_strategy_function


def main():
    logging.getLogger().setLevel(logging.WARNING)

        ## ---------ADJUST THIS---------------

    ticker_lst = ['AES', 'ALB', 'AOS', 'APA', 'BEN', 'BXP', 'CPB',
                  'DVA', 'FFIV', 'FRT', 'HII', 'HRL', 'HSIC', 'INCY',
                  'MHK', 'NWSA', 'PNW', 'RL', 'TAP', 'WYNN']


    testing_year = "2019"
    params_year = "2018"
    order_type = 'combined'
    ret_type = "log_ret_ex"
    params_file_name = "all_log_ret_ex_2018_v2.csv"

    delta = '2min'
    pos_threshold = 0
    neg_threshold = 0
    result_path = "/nfs/home/jingt/dissertation-iceberg/data/output_results/strategy_results"
    file_name = "all_strat_single_combined_log_ret_ex_update_2min.csv"

    ## ------------------------------------

    df_ticker_lst = []

    # Check if files exist
    params_folder_path = f"/nfs/home/jingt/dissertation-iceberg/data/output_results/{params_year}"
    model_path = "/nfs/home/jingt/dissertation-iceberg/data/output_folder"

    params_file_path = os.path.join(params_folder_path, params_file_name)
    params_df = pd.read_csv(params_file_path)

    # Check all tickers are in params_df and files exist

    duplicates = params_df.duplicated(subset=['ticker', 'timeframe'])

    # Display the duplicated rows
    if duplicates.any():
        print("Duplicate rows based on 'ticker' and 'timeframe':", flush=True)
        print(params_df[duplicates], flush=True)
    else:
        print("No duplicates found based on 'ticker' and 'timeframe'.", flush=True)


    for ticker in ticker_lst:
        file_path_test = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{testing_year}-01-01_{testing_year}-12-31_10.7z"

        if not os.path.exists(file_path_test):
            print(f"The file {file_path_test} does not exist.", flush=True)
        
        if ticker not in params_df['ticker'].unique():
            print(f"{ticker} not in params dataframe!", flush=True)
        
    for ticker in ticker_lst:
        start = time.time()
        print(f"Calculating for ticker {ticker}", flush=True)
        df_final, result_final_unweighted, result_final_weighted = combined_strategy_function(ticker, delta, order_type, ret_type, model_path, 
                                                pos_threshold=0, neg_threshold=0, weighted=False, momentum=False, year=2019, 
                                                use_update_strategy=True, params_df=None)

        df_final['final_PnL_weighted'] = result_final_weighted
        df_final['final_PnL_unweighted'] = result_final_unweighted
        df_ticker_lst.append(df_final)




        df_ticker_all = pd.concat(df_ticker_lst)

        save_dataframe_to_folder(df_ticker_all, result_path, file_name)

        print(f"{time.time() - start:.3f} seconds elapsed", flush=True)

        print(f"Done for ticker {ticker}!", flush=True)


if __name__ == '__main__':
    main()

    
        