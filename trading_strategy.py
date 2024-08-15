import pandas as pd
import numpy as np
from chunk_pipeline import order_imbalance_calc

def OI_signals_daily_ticker(params_df, ticker, date, delta, order_type, ret_type, model_path):
    # For a particular day
    # For each ticker, predict direction and calculate OI signals for ticker
    # Output dataframe of OI signals for particular day for all tickers
    # Output should be time | ticker | OI Signal | fut_log_ret_ex (next bucket)

    # Obtain Order Imbalance for ticker in specified date
    year = date[:4] # Check if this is the correct form
    test_file_path = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{year}-01-01_{year}-12-31_10.7z"
    model_name = f"xgboos_{ticker}.json"

    OI_df = order_imbalance_calc(test_file_path, delta_lst=[delta], model=None,
                                 model_path=model_path, model_name=model_name,
                                 order_type=order_type, specific_date=date)

    OI_df = OI_df[delta]

    if ret_type == "log_ret_ex":
        fret_df = OI_df["fut_log_ret_ex"]
    else:
        print("Not Implemented!")
        return

    if order_type == "combined":
        


    
    # Calculate OI signal



def signal_ranking_daily(ticker_lst, date, delta, order_type, model_path):
    # Call OI signal calculation
    # Rank signals fot tickers for each delta bucket
    # Calculate PnL for day of entire strategy and other statistics
    # Calculate PnL for each ticker
    for ticker in ticker_lst:

    


def process_signals(ticker_lst, delta_lst, test_file_path):
    # Loops through each day
    # Feeds data into OI signals daily
    # Feeds df into signal_ranking_daily
    # Calculate overall strategy statistics
    pass