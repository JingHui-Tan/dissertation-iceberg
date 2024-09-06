from chunk_pipeline import *
from prediction_ML_pipeline import save_dataframe_to_folder
import time
import os
from trading_strategy import combined_strategy_function, portfolio_update_signals, ClOp_signal


def main():
    logging.getLogger().setLevel(logging.WARNING)

        ## ---------ADJUST THIS---------------

    ticker_lst = ['AES', 'ALB', 'AOS', 'APA', 'BEN', 'BXP', 'CPB',
                  'DVA', 'FFIV', 'FRT', 'HII', 'HRL', 'HSIC', 'INCY',
                  'MHK', 'NWSA', 'PNW', 'RL', 'TAP', 'WYNN']



    testing_year = "2019"
    order_type = 'combined'
    file_order = 'comb'

    percentile = 0.05

    #params_path = "/nfs/home/jingt/dissertation-iceberg/data/output_regression/pred_ClOp_all_ff_p2.csv"
    params_path = f"/nfs/home/jingt/dissertation-iceberg/data/output_regression/pred_ClOp_ff_{file_order}.csv"
    result_path = "/nfs/home/jingt/dissertation-iceberg/data/trading_strat"
    file_name_results = f"results_ClOp_{order_type}.csv"
    file_name_counts = f"counts_ClOp_{order_type}.csv"
    pnl_file_name = f'pnl_ClOp_{order_type}.csv'


    ## ------------------------------------

    df_ticker_lst = []

    # Check if files exist
    model_path = "/nfs/home/jingt/dissertation-iceberg/data/output_folder"



    for ticker in ticker_lst:
        file_path_test = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{testing_year}-01-01_{testing_year}-12-31_10.7z"

        if not os.path.exists(file_path_test):
            print(f"The file {file_path_test} does not exist.", flush=True)

    start = time.time()
    ClOp_signal(params_path=params_path, ticker_lst=ticker_lst, order_type=order_type, model_path=model_path, ret_type='ClOp', delta='daily',
                                                           percentile=0.05)



    print(f"{time.time() - start:.3f} seconds elapsed", flush=True)
    print('Done', flush=True)

if __name__ == '__main__':
    main()

    
        