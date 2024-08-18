import pandas as pd
import numpy as np
from chunk_pipeline import order_imbalance_calc
import exchange_calendars as mcal
import pickle
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score




def OI_signals_daily_ticker(params_df, ticker, date, delta, order_type, ret_type, model_path):
    # For a particular day
    # For each ticker, predict direction and calculate OI signals for ticker
    # Output dataframe of OI signals for particular day for all tickers
    # Output should be time | ticker | OI Signal | fut_log_ret_ex (next bucket)

    # Obtain Order Imbalance for ticker in specified date
    year = date[:4] # Check if this is the correct form
    test_file_path = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{year}-01-01_{year}-12-31_10.7z"
    model_name = f"xgboost_{ticker}.json"

    # Calculate OI imbalance
    OI_df = order_imbalance_calc(test_file_path, delta_lst=[delta], model=None,
                                 model_path=model_path, model_name=model_name,
                                 order_type=order_type, specific_date=date)

    OI_df = OI_df[delta]
    params_df_ticker = params_df[(params_df['ticker'] == ticker) & (params_df['timeframe'] == delta)]

    # Calculate OI Signal
    if ret_type == "log_ret_ex":
        signal_df = OI_df[["datetime_bins", "fut_log_ret_ex"]]
        signal_df['ticker'] = ticker

    else:
        print("Not Implemented!")
        return

    if order_type == "combined":
        signal_df['signal'] = (float(params_df_ticker['intercept']) 
                              + float(params_df_ticker['params_vis']) * OI_df['order_imbalance_vis'] 
                              + float(params_df_ticker['params_hid']) * OI_df['order_imbalance_hid'])
    
    elif order_type == 'hid':
        signal_df['signal'] = (float(params_df_ticker['intercept'])
                               + float(params_df_ticker['params_hid']) * OI_df['order_imbalance_hid'])
    
    elif order_type == 'comb_iceberg':
        signal_df['signal'] = (float(params_df_ticker['intercept']) 
                              + float(params_df_ticker['params_vis']) * OI_df['order_imbalance_vis'] 
                              + float(params_df_ticker['params_ib']) * OI_df['order_imbalance_ib'] 
                              + float(params_df_ticker['params_hid']) * OI_df['order_imbalance_hid'])

    return signal_df


def portfolio_construction_daily(params_df, ticker_lst, date, delta, 
                                 order_type, ret_type, model_path, percentile=0.2):
    # Call OI signal calculation
    # Rank signals fot tickers for each delta bucket
    # Calculate PnL for day of entire strategy and other statistics
    # Calculate PnL for each ticker
    signal_df_all = []

    # Merge all signal dataframes for each ticker
    for ticker in ticker_lst:
        signal_df_ticker = OI_signals_daily_ticker(params_df, ticker, date, delta, 
                                                   order_type, ret_type, model_path)
        signal_df_all.append(signal_df_ticker)
    
    signal_df_all = pd.concat(signal_df_all).reset_index(drop=True)

    # Rank the signals for each datetime_bin
    signal_df_all['signal_rank'] = signal_df_all.groupby('datetime_bins')['signal'].rank(method='first', ascending=False)

    # Calculate the thresholds for top and bottom percentages
    top_percentile = int(percentile * len(ticker_lst))

    top_signals = signal_df_all[signal_df_all['signal_rank'] <= top_percentile]
    top_signals = top_signals[top_signals[['signal']] > 0]
    bottom_signals = signal_df_all[signal_df_all['signal_rank'] >= len(ticker_lst) - top_percentile]
    bottom_signals = bottom_signals[bottom_signals[['signal']] < 0]

    # Group by 'ticker' and count the occurrences for top signals
    top_counts = top_signals.groupby('ticker').count()[['signal']]
    top_counts.rename(columns={'signal': 'top_counts'}, inplace=True)

    # Group by 'ticker' and count the occurrences for bottom signals
    bottom_counts = bottom_signals.groupby('ticker').count()[['signal']]
    bottom_counts.rename(columns={'signal': 'bottom_counts'}, inplace=True)

    df_counts = pd.merge(top_counts, bottom_counts, left_index=True, right_index=True)

    top_sum  = top_signals['fut_log_ret_ex'].sum()
    bottom_sum = bottom_signals['fut_log_ret_ex'].sum()

    result = top_sum - bottom_sum
    
    number_runs = len(signal_df_ticker)

    return df_counts, result, number_runs


def process_signals(params_df, ticker_lst, delta, ret_type, model_path, order_type, percentile=0.2, year=2019):
    # Loops through each day
    # Feeds data into OI signals daily
    # Feeds df into signal_ranking_daily
    # Calculate overall strategy statistics

    total_runs_lst = []
    PnL_results_lst = []
    df_counts_lst = []

    nasdaq = mcal.get_calendar('XNYS')  # XNYS is often used for NASDAQ

    # Get the trading schedule for the entire year
    trading_schedule = nasdaq.sessions_in_range(f'{year}-01-01', f'{year}-12-31')

    # Convert to a list of strings
    trading_days_lst = trading_schedule.strftime('%Y-%m-%d').tolist()

    for date in trading_days_lst:
        df_counts, result, number_runs = portfolio_construction_daily(params_df, ticker_lst, date, delta, 
                                                                      order_type, ret_type, model_path, 
                                                                      percentile=percentile)

        total_runs_lst.append(number_runs)
        PnL_results_lst.append(result)
        df_counts_lst.append(df_counts)

    return total_runs_lst, PnL_results_lst, df_counts_lst



def agg_ind_ticker_strat(params_df, ticker_lst, delta, order_type, ret_type, model_path, 
                         pos_threshold, neg_threshold, weighted=False, year=2019):
    df_ticker_lst = []
    for ticker in ticker_lst:
        df_ticker, result_final_ticker = signal_single_ticker_strat(params_df, ticker, delta, order_type, ret_type, 
                                                                    model_path, pos_threshold, neg_threshold, 
                                                                    weighted=weighted, year=year)
        df_ticker['final_PnL'] = result_final_ticker
        df_ticker_lst.append(df_ticker)
    
    df_ticker_all = pd.concat(df_ticker_lst)

    return df_ticker_all




def update_strategy_single_daily(ticker, date, delta, order_type, ret_type, model_path, momentum=False):
    # Obtain Order Imbalance for ticker in specified date
    year = date[:4] # Check if this is the correct form
    test_file_path = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{year}-01-01_{year}-12-31_10.7z"
    model_name = f"xgboost_{ticker}.json"

    OI_df_lst = []

    # Define the exchange calendar (XNYS is typically used for NYSE)
    nyse = mcal.get_calendar('XNYS')

    # Get all trading sessions up to the given date
    all_sessions = nyse.sessions_in_range(pd.Timestamp('2015-01-01'), pd.Timestamp(date))

    # Get the last 5 trading sessions before the given date
    five_days_before = all_sessions[-6:-1]  # Exclude the given date itself

    # Convert to a list of strings
    five_days_before_list = five_days_before.strftime('%Y-%m-%d').tolist()


    for trading_date in five_days_before_list:
        # Calculate OI imbalance
        OI_df = order_imbalance_calc(test_file_path, delta_lst=[delta], model=None,
                                    model_path=model_path, model_name=model_name,
                                    order_type=order_type, specific_date=trading_date)
        
        OI_df = OI_df[delta]
        OI_df_lst.append(OI_df)
    
    OI_df_full = pd.concat(OI_df_lst)

    OI_df_curr = order_imbalance_calc(test_file_path, delta_lst=[delta], model=None,
                                    model_path=model_path, model_name=model_name,
                                    order_type=order_type, specific_date=date)
    OI_df_curr = OI_df_curr[delta]

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

    X_coef = coefficients_dict[order_type]


    if ret_type == "log_ret_ex":
        output = "fut_log_ret_ex"
        if momentum:
            X_coef += ['log_ret_ex']
    
    X_train = OI_df_full[X_coef].fillna(0).replace(-np.inf, 0).replace(np.inf, 0)
    y_train = OI_df_full[output].fillna(0).replace(-np.inf, 0).replace(np.inf, 0)

    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)

    signal = model.predict(OI_df_curr[X_coef])
    
    signal_df = OI_df_curr[[output]]
    signal_df["signal"] = signal

    return signal_df



def combined_strategy_function(ticker, delta, order_type, ret_type, model_path, 
                               pos_threshold=0, neg_threshold=0, weighted=False, momentum=False, year=2019, 
                               use_update_strategy=True, params_df=None):

    result_final = 0
    result_lst = []

    nasdaq = mcal.get_calendar('XNYS')  # XNYS is often used for NASDAQ
    # Get the trading schedule for the entire year
    trading_schedule = nasdaq.sessions_in_range(f'{year}-01-01', f'{year}-12-31')
    # Convert to a list of strings
    trading_days_lst = trading_schedule.strftime('%Y-%m-%d').tolist()

    for date in trading_days_lst[6:] if use_update_strategy else trading_days_lst:
        if use_update_strategy:
            signal_day = update_strategy_single_daily(ticker, date, delta, order_type, ret_type, model_path, momentum=momentum)
        else:
            signal_day = OI_signals_daily_ticker(params_df, ticker, date, delta, order_type, ret_type, model_path)

        if not weighted:
            positive_sum = signal_day[signal_day['signal'] > pos_threshold]['fut_log_ret_ex'].sum()
            negative_sum = signal_day[signal_day['signal'] < neg_threshold]['fut_log_ret_ex'].sum()
        else:
            positive_sum = (signal_day[signal_day['signal'] > 0]['fut_log_ret_ex'] * signal_day[signal_day['signal'] > 0]['signal'].abs()).sum()
            negative_sum = (signal_day[signal_day['signal'] < 0]['fut_log_ret_ex'] * signal_day[signal_day['signal'] < 0]['signal'].abs()).sum()

        result = positive_sum - negative_sum
        result_final += result
        result_lst.append(result)

    df = pd.DataFrame([result_lst], columns=trading_days_lst[6:] if use_update_strategy else trading_days_lst)
    df.insert(0, "Ticker", ticker)
    
    return df, result_final
