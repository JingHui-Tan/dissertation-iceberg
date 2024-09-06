import pandas as pd
import numpy as np
from chunk_pipeline import order_imbalance_calc
import exchange_calendars as mcal
import pickle
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from order_imbalance import get_datetime_bins
from prediction_ML_pipeline import save_dataframe_to_folder
import os
from datetime import datetime



def OI_signals_daily_ticker(params_df, ticker, date, delta, order_type, ret_type, model_path):
    # For a particular day and ticker, predict direction and calculate OI signals for ticker

    # Obtain Order Imbalance for ticker in specified date
    year = date[:4]
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

    if ret_type == "ClCl" or ret_type == 'ClOp' or ret_type == 'adjClOp':
        signal_df = OI_df[['datetime_bins', f'fret_{ret_type}']]

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
    # Rank signals fot tickers for each delta bucket and calculate PnL
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
    # Loops through each day, feeds data into OI signals daily, Calculate overall strategy statistics

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






def update_strategy_single_daily(ticker, date, delta, order_type, ret_type, model_path, prev_days=5, momentum=False):
    # Calculate signals for the current day for specified ticker using coefficients obtained from previous trading days
 
    year = date[:4]
    test_file_path = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{year}-01-01_{year}-12-31_10.7z"
    model_name = f"xgboost_{ticker}.json"

    OI_df_lst = []

    if ret_type != 'ClOp' and ret_type != 'ClCl' and ret_type != 'adjClOp':
        # Define the exchange calendar (XNYS is typically used for NYSE)
        nyse = mcal.get_calendar('XNYS')

        # Get all trading sessions up to the given date
        all_sessions = nyse.sessions_in_range(pd.Timestamp('2017-01-01'), pd.Timestamp(date))

        # Get the last few trading sessions before the given date
        prev_days += 1
        days_before = all_sessions[-prev_days:-1]  # Exclude the given date itself

        # Convert to a list of strings
        days_before_list = days_before.strftime('%Y-%m-%d').tolist()

    if ret_type == "ClOp" or ret_type == "ClCl" or ret_type == 'adjClOp':
        # List to store filenames without the .csv.gz extension
        file_names_without_extension = []
        fret_folder = "/nfs/home/jingt/dissertation-iceberg/data/fret_folder"
        # Iterate over all the files in the folder
        for file_name in os.listdir(fret_folder):
            # Check if the file has the .csv.gz extension
            if file_name.endswith('.csv.gz'):
                # Remove the .csv.gz extension
                name_without_extension = file_name[:-7]  # Removes the last 7 characters (.csv.gz)
                file_names_without_extension.append(name_without_extension)
        
        # Define the exchange calendar (XNYS is typically used for NYSE)
        nyse = mcal.get_calendar('XNYS')

        # Get all trading sessions up to the given date
        all_sessions = nyse.sessions_in_range(pd.Timestamp('2017-01-01'), pd.Timestamp(date))

        # Get the last few trading sessions before the given date
        prev_days += 1
        days_before = all_sessions[-prev_days:-1]  # Exclude the given date itself

        # Convert to a list of strings
        days_before_list_all = days_before.strftime('%Y-%m-%d').tolist()
        
        days_before_list = []
        for day in days_before_list_all:
            if day in file_names_without_extension:
                days_before_list.append(day)
                
        print(days_before_list)
    for trading_date in days_before_list:
        # Calculate OI imbalance
        OI_df = order_imbalance_calc(test_file_path, delta_lst=[delta], model=None,
                                    model_path=model_path, model_name=model_name,
                                    order_type=order_type, specific_date=trading_date, ticker=ticker)
        
        OI_df = OI_df[delta]
        OI_df_lst.append(OI_df)
    
    OI_df_full = pd.concat(OI_df_lst)

    # Calculate OI for current day
    OI_df_curr = order_imbalance_calc(test_file_path, delta_lst=[delta], model=None,
                                    model_path=model_path, model_name=model_name,
                                    order_type=order_type, specific_date=date, ticker=ticker)
    OI_df_curr = OI_df_curr[delta]

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

    # Fit linear regression model on previous days
    X_coef = coefficients_dict[order_type]

    if ret_type == "log_ret_ex":
        output = "fut_log_ret_ex"
        if momentum:
            X_coef += ['log_ret_ex']
        
    elif ret_type == "ClOp" or ret_type == "ClCl" or ret_type=='adjClOp':
        output = f'fret_{ret_type}'
        if momentum:
            X_coef += [f'{ret_type}', 'SMB', 'HML', 'RF'] # Based on FF three-factor model
    
    X_train = OI_df_full[X_coef].fillna(0).replace(-np.inf, 0).replace(np.inf, 0)
    y_train = OI_df_full[output].fillna(0).replace(-np.inf, 0).replace(np.inf, 0)

    # Fit linear regression
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict fret for current days as signal
    signal = model.predict(OI_df_curr[X_coef].fillna(0).replace(-np.inf, 0).replace(np.inf, 0))
    signal_df = OI_df_curr[[output]]
    signal_df["signal"] = signal
    return signal_df


def get_trading_days(ret_type, year=2019):
    # Get training day for current year, filter ClOp trading days based on fret file

    if ret_type == 'log_ret_ex':
        if year == 2019:
            nasdaq = mcal.get_calendar('XNYS')  # XNYS is often used for NASDAQ
            # Get the trading schedule for the entire year
            trading_schedule = nasdaq.sessions_in_range(f'{year}-01-01', f'{year}-12-31')
            # Convert to a list of strings
            trading_days_lst = trading_schedule.strftime('%Y-%m-%d').tolist()
            return trading_days_lst

    if ret_type == "ClCl" or ret_type == 'ClOp' or ret_type =='adjClOp':
        nasdaq = mcal.get_calendar('XNYS')  # XNYS is often used for NASDAQ
        # Get the trading schedule for the entire year
        trading_schedule = nasdaq.sessions_in_range(f'{year}-01-01', f'{year}-12-31')
        # Convert to a list of strings
        trading_days_lst = trading_schedule.strftime('%Y-%m-%d').tolist()
        
        # List to store filenames without the .csv.gz extension
        file_names_without_extension = []
        fret_folder = "/nfs/home/jingt/dissertation-iceberg/data/fret_folder"
        # Iterate over all the files in the folder
        for file_name in os.listdir(fret_folder):
            # Check if the file has the .csv.gz extension
            if file_name.endswith('.csv.gz'):
                # Remove the .csv.gz extension
                name_without_extension = file_name[:-7]  # Removes the last 7 characters (.csv.gz)
                file_names_without_extension.append(name_without_extension)
        
        trading_days_lst_filt = []
        for day in trading_days_lst:
            if day in file_names_without_extension:
                trading_days_lst_filt.append(day)

        return trading_days_lst_filt





def combined_strategy_function(ticker, delta, order_type, ret_type, model_path, 
                               pos_threshold=0, neg_threshold=0, weighted=False, momentum=False, year=2019, 
                               prev_days=5, use_update_strategy=True, params_df=None):
    # Call specified signal strategy and compute daily PnL
    result_lst_unweighted = []
    result_lst_weighted = []

    if ret_type == 'log_ret_ex':
        returns = 'fut_log_ret_ex'
    
    elif ret_type == 'ClOp' or ret_type == 'ClCl' or ret_type =='adjClOp':
        returns = f'fret_{ret_type}'
    
    # Get trading days for specified year
    trading_days_lst = get_trading_days(ret_type, year=year)

    # Calculate signals for each trading day
    for date in trading_days_lst[6:] if use_update_strategy else trading_days_lst:
        if use_update_strategy:
            signal_day = update_strategy_single_daily(ticker, date, delta, order_type, ret_type, model_path, prev_days=prev_days, momentum=momentum)
        else:
            signal_day = OI_signals_daily_ticker(params_df, ticker, date, delta, order_type, ret_type, model_path)

        # Calculate PnL

        # Unweighted calculations
        positive_sum_unweighted = signal_day[signal_day['signal'] > pos_threshold][returns].sum()
        negative_sum_unweighted = signal_day[signal_day['signal'] < neg_threshold][returns].sum()
        result_unweighted = positive_sum_unweighted - negative_sum_unweighted
        result_lst_unweighted.append(result_unweighted)

        # Weighted calculations
        positive_sum_weighted = (signal_day[signal_day['signal'] > 0][returns] * signal_day[signal_day['signal'] > 0]['signal'].abs()).sum()
        negative_sum_weighted = (signal_day[signal_day['signal'] < 0][returns] * signal_day[signal_day['signal'] < 0]['signal'].abs()).sum()
        result_weighted = positive_sum_weighted - negative_sum_weighted
        result_lst_weighted.append(result_weighted)

    # Create DataFrames for unweighted and weighted results
    df_unweighted = pd.DataFrame([result_lst_unweighted], columns=trading_days_lst[6:] if use_update_strategy else trading_days_lst)
    df_unweighted.insert(0, "Ticker", ticker)
    df_unweighted.insert(1, "Type", "Unweighted")

    df_weighted = pd.DataFrame([result_lst_weighted], columns=trading_days_lst[6:] if use_update_strategy else trading_days_lst)
    df_weighted.insert(0, "Ticker", ticker)
    df_weighted.insert(1, "Type", "Weighted")

    # Concatenate the unweighted and weighted DataFrames
    df_final = pd.concat([df_unweighted, df_weighted], ignore_index=True)

    # Compute the final result for both weighted and unweighted
    result_final_unweighted = sum(result_lst_unweighted)
    result_final_weighted = sum(result_lst_weighted)

    # Return the final DataFrame and the final result for both types
    return df_final, result_final_unweighted, result_final_weighted



def portfolio_update_signals(ticker_lst, delta, order_type, ret_type, model_path, 
                             save_file_path, result_file_name, count_file_name, pnl_file_name, params_path=None, version=False, 
                             prev_days=3, percentile=0.2, momentum=False, year=2019):
    # Run portfolio strategy for specified list of tickers, timeframe, return type
    result_all_lst = []
    df_counts_all_lst = []
    df_pnl_ticker_lst = []
    top_percentile = int(percentile * len(ticker_lst))

    if ret_type == 'log_ret_ex':
        target = 'fut_log_ret_ex'
    
    elif ret_type == 'ClCl' or ret_type == 'ClOp' or ret_type == 'adjClOp':
        target = f'fret_{ret_type}'

    
    trading_days_lst = get_trading_days(ret_type, year=year)

    for date in trading_days_lst[6:]:
        signal_day = []

        start_date = date
        end_date = date

        if not delta == 'daily':
            start_datetime = pd.Timestamp.combine(pd.to_datetime(start_date), pd.Timestamp("10:00").time())
            start_datetime = start_datetime + pd.Timedelta(delta)
            end_datetime = pd.Timestamp.combine(pd.to_datetime(end_date), pd.Timestamp("15:30").time())

            # Generate the full range of datetime bins with the specified frequency
            full_range = pd.date_range(start=start_datetime, end=end_datetime, freq=delta)

            # Create the DataFrame with the datetime bins
            full_bins = pd.DataFrame(full_range, columns=['datetime_bins'])

        if delta == 'daily':
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
            start_date_dt = start_date_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            full_bins = pd.DataFrame([start_date_dt], columns=['datetime_bins'])

        # Compute signal for each ticker for current day
        for ticker in ticker_lst:
            signal_df_ticker_daily = update_strategy_single_daily(ticker, date, delta, order_type, 
                                                                ret_type, model_path, prev_days=prev_days, momentum=momentum)
            signal_df_ticker_daily['ticker'] = ticker
            signal_df_ticker_daily['datetime_bins'] = full_bins['datetime_bins']  
            signal_day.append(signal_df_ticker_daily)

        df_signal_day = pd.concat(signal_day).reset_index(drop=True)

        # Group by 'datetime_bins' and compute rank of signal
        df_signal_day['signal_rank'] = df_signal_day.groupby('datetime_bins')['signal'].rank(method='first', ascending=False)

        # Obtain top and bottom signals for portfolio strategy
        top_signals = df_signal_day[df_signal_day['signal_rank'] <= top_percentile]
        top_signals = top_signals[top_signals['signal'] > 0]
        bottom_signals = df_signal_day[df_signal_day['signal_rank'] > len(ticker_lst) - top_percentile]
        bottom_signals = bottom_signals[bottom_signals['signal'] < 0]


        # Group by 'ticker' and count the occurrences for top signals
        top_counts = top_signals.groupby('ticker').count()[['signal']]
        top_counts.rename(columns={'signal': 'top_counts'}, inplace=True)

        # Group by 'ticker' and count the occurrences for bottom signals
        bottom_counts = bottom_signals.groupby('ticker').count()[['signal']]
        bottom_counts.rename(columns={'signal': 'bottom_counts'}, inplace=True)

        df_counts = pd.merge(top_counts, bottom_counts, left_index=True, right_index=True, how='outer')
        df_counts = df_counts.fillna(0)
        df_counts.reset_index(inplace=True)
        df_counts['date'] = date
    
        # Obtain PnL of top and bottom signals
        top_sum  = top_signals[target].sum()
        bottom_sum = bottom_signals[target].sum()

        # Compute ticker level PnL
        top_sum_by_ticker = top_signals.groupby('ticker')[target].sum().reset_index()
        top_sum_by_ticker.rename(columns={target: 'top_pnl'}, inplace=True)

        bottom_sum_by_ticker = bottom_signals.groupby('ticker')[target].sum().reset_index()
        bottom_sum_by_ticker.rename(columns={target: 'bottom_pnl'}, inplace=True)
        pnl_by_ticker = pd.merge(top_sum_by_ticker, bottom_sum_by_ticker, left_on='ticker', right_on='ticker', how='outer')
        pnl_by_ticker.fillna(0, inplace=True)
        pnl_by_ticker['date'] = date

        # Compute final PnL and number of runs
        result = top_sum - bottom_sum
        number_runs = len(signal_df_ticker_daily)

        print(f"PnL for {date}: {result}", flush=True)
        print(f"number of runs for {date}: {number_runs}", flush=True)
        with pd.option_context('display.max_rows', 100, 'display.max_columns', 10):
            print(df_counts, flush=True)
            print(pnl_by_ticker, flush=True)

        # Create dataframes for future analysis
        result_df = pd.DataFrame({'date': [date], 'PnL': [result], 'no. runs': [number_runs]})

        result_all_lst.append(result_df)
        df_counts_all_lst.append(df_counts)
        df_pnl_ticker_lst.append(pnl_by_ticker)

        df_result_all = pd.concat(result_all_lst)
        df_counts_all = pd.concat(df_counts_all_lst)
        df_pnl_all = pd.concat(df_pnl_ticker_lst)

        save_dataframe_to_folder(df_result_all, save_file_path, result_file_name)
        save_dataframe_to_folder(df_counts_all, save_file_path, count_file_name)
        save_dataframe_to_folder(df_pnl_all, save_file_path, pnl_file_name)



        with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
            print(df_result_all, flush=True)
    
    return df_result_all, df_counts_all, df_pnl_all




def ClOp_signal(params_path, ticker_lst, order_type, model_path, ret_type='ClOp', delta='daily',
                percentile=0.05):
    # Compute signals for ClOp daily strategy
    signal_lst = []
    for ticker in ticker_lst:
        # Obtain Order Imbalance for ticker in specified date
        test_file_path = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{2019}-01-01_{2019}-12-31_10.7z"
        model_name = f"xgboost_{ticker}.json"
        params_df = pd.read_csv(params_path)

        params_df_ticker = params_df[(params_df['ticker'] == ticker) & (params_df['timeframe'] == delta)]
        print(params_df_ticker)

        # Calculate OI imbalance
        OI_df = order_imbalance_calc(test_file_path, delta_lst=[delta], model=None,
                                    model_path=model_path, model_name=model_name,
                                    order_type=order_type, ticker=ticker)

        OI_df = OI_df[delta]


        signal_df = OI_df[['datetime_bins', f'fret_ClOp']]

        # Compute signal based on order_type specified
        if order_type == "combined":
            signal_df['signal'] = (float(params_df_ticker['intercept'].iloc[0]) 
                                + float(params_df_ticker['params_vis'].iloc[0]) * OI_df['order_imbalance_vis'] 
                                + float(params_df_ticker['params_hid'].iloc[0]) * OI_df['order_imbalance_hid'])
        
        elif order_type == 'all':
            signal_df['signal'] = (float(params_df_ticker['intercept'].iloc[0]) 
                                + float(params_df_ticker['params_all'].iloc[0]) * OI_df['order_imbalance_all'])
        
        elif order_type == 'hid':
            signal_df['signal'] = (float(params_df_ticker['intercept'].iloc[0])
                                + float(params_df_ticker['params_hid'].iloc[0]) * OI_df['order_imbalance_hid'])
        
        elif order_type == 'comb_iceberg':
            signal_df['signal'] = (float(params_df_ticker['intercept'].iloc[0]) 
                                + float(params_df_ticker['params_vis'].iloc[0]) * OI_df['order_imbalance_vis'] 
                                + float(params_df_ticker['params_ib'].iloc[0]) * OI_df['order_imbalance_ib'] 
                                + float(params_df_ticker['params_hid'].iloc[0]) * OI_df['order_imbalance_hid'])

        signal_df['signal'] += (float(params_df_ticker['params_SMB'].iloc[0]) * OI_df['SMB'] 
                                + float(params_df_ticker['params_HML'].iloc[0]) * OI_df['HML'] 
                                + float(params_df_ticker['params_RF'].iloc[0]) * OI_df['RF'] 
                                + float(params_df_ticker['params_CMA'].iloc[0]) * OI_df['CMA']
                                + float(params_df_ticker['params_RMW'].iloc[0]) * OI_df['RMW']  
                                )
        signal_df['ticker'] = ticker

        folder_path = f'/nfs/home/jingt/dissertation-iceberg/data/trading_strat/ClOp_{order_type}'
        file_name = f'ClOp_{ticker}.csv'
        save_dataframe_to_folder(signal_df, folder_path, file_name)

        break

