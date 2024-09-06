from chunk_pipeline import *
from prediction_ML_pipeline import save_dataframe_to_folder
import time
import os
from trading_strategy import combined_strategy_function, portfolio_update_signals


def main():
    logging.getLogger().setLevel(logging.WARNING)

        ## ---------ADJUST THIS---------------

    ticker_lst = ['AES', 'ALB', 'AOS', 'APA', 'BEN', 'BXP', 'CPB',
                  'DVA', 'FFIV', 'FRT', 'HII', 'HRL', 'HSIC', 'INCY',
                  'MHK', 'NWSA', 'PNW', 'RL', 'TAP', 'WYNN']

# Top tickers for 2 minutes and 30 minutes portfolios
# 2min
# All - ['INCY', 'NWSA', "BXP', 'APA', 'TAP']
# iceberg - ['BXP', 'INCY', 'DVA', 'APA', 'TAP']
# combined - ['APA', 'NWSA', 'DVA', 'BXP', 'TAP']

# 30min
# All - ['HSIC', 'ALB', 'MHK', 'BXP', 'INCY']
# iceberg - ['NWSA', 'HSIC', 'ALB', 'BEN', 'BXP']
# comb - ['HSIC', 'ALB', 'MHK', 'BXP', 'INCY']

    testing_year = "2019"
    order_type = 'combined'
    ret_type = "log_ret_ex"
    momentum = True

    delta = '30min'
    prev_days = 10
    percentile = 1/5
    pos_threshold = 0
    neg_threshold = 0
    result_path = "/nfs/home/jingt/dissertation-iceberg/data/trading_strat"
    file_name_results = "results_30min_comb_red.csv"
    file_name_counts = "counts_30min_comb_red.csv"
    pnl_file_name = 'pnl_30min_comb_red.csv'


    ## ------------------------------------

    df_ticker_lst = []

    # Check if files exist
    model_path = "/nfs/home/jingt/dissertation-iceberg/data/output_folder"



    for ticker in ticker_lst:
        file_path_test = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{testing_year}-01-01_{testing_year}-12-31_10.7z"

        if not os.path.exists(file_path_test):
            print(f"The file {file_path_test} does not exist.", flush=True)

        
    start = time.time()
    df_result_all, df_counts_all, df_pnl_all = portfolio_update_signals(ticker_lst, delta, order_type, ret_type, model_path, 
                                                            save_file_path=result_path, result_file_name=file_name_results, count_file_name=file_name_counts,
                                                            pnl_file_name=pnl_file_name, prev_days=prev_days, percentile=percentile, momentum=momentum, year=2019)



    print(f"{time.time() - start:.3f} seconds elapsed", flush=True)
    print('Done', flush=True)

if __name__ == '__main__':
    main()

    
        