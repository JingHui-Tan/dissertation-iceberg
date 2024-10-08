
'''Compute different versions of conditional Order Imbalance on message and orderbook dataframe'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.formula.api as smf
import yfinance as yf
from datetime import timedelta
import logging
from prediction_ML_pipeline import save_dataframe_to_folder
import os
import gzip
import time


## Process:
## We fit order_imbalance and then remove outliers based on 90% quantile (for log returns and future log returns)

logging.getLogger().setLevel(logging.WARNING)


def iceberg_tag(df, ib_delta):
    # Add a column tagging whether trade is an iceberg execution
    event_type_4 = df[df['event_type'] == 4]
    event_type_1 = df[df['event_type'] == 1]

    # Sort both DataFrames by datetime
    event_type_4 = event_type_4.sort_values(by='datetime')
    event_type_1 = event_type_1.sort_values(by='datetime')

    # Merge based on small delta timeframe
    merged = pd.merge_asof(
        event_type_4,
        event_type_1,
        on='datetime',
        by=['ticker', 'price'],
        direction='forward',
        tolerance=pd.Timedelta(ib_delta),
        suffixes=('', '_event_1')  
    )
    event_1_within_delta = merged.loc[merged['direction_event_1'].notna(), 'datetime']
    event_type_4['iceberg'] = 0
    event_type_4.loc[event_type_4.index.get_level_values('datetime').isin(event_1_within_delta), 'iceberg'] = 1

    return event_type_4


def fetch_ticker_data_with_retry(ticker, date, retries=10, delay=5, include_prev=False):
    # Get ticker and SPY data from yfinance with retries
    if not include_prev:
        for i in range(retries):
            try:
                data = yf.download(ticker, start=date, end=date + timedelta(days=7))
                if data.empty:
                    raise ValueError(f"No data found for {ticker}")
                return data
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    print(f"Failed to retrieve data for {ticker} after {retries} attempts.")
                    return None
    
    else:
        for i in range(retries):
            try:
                data = yf.download([ticker, 'SPY'], start=date - timedelta(days=7), end=date + timedelta(days=7))
                if data.empty:
                    raise ValueError(f"No data found for {ticker}")
                return data
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    print(f"Failed to retrieve data for {ticker} after {retries} attempts.")
                    return None
        


def calculate_log_returns(grouped, delta, ticker=None):
    current_date = grouped['datetime_bins'][0].date()

    # Calculate log returns for intraday timeframe
    if delta != 'daily':
        grouped['log_ret'] = np.log(grouped['last_midprice']) - np.log(grouped['first_midprice'])
        grouped['fut_log_ret'] = grouped['log_ret'].shift(-1)
        grouped['weighted_log_ret'] = np.log(grouped['last_weighted_mp']) - np.log(grouped['first_weighted_mp'])
        grouped['fut_weighted_log_ret'] = grouped['weighted_log_ret'].shift(-1)

        # Get market excess returns
        SPY_data_loc = f"/nfs/home/jingt/dissertation-iceberg/data/SPY_data/SPY_{current_date}.csv"
        SPY_data = pd.read_csv(SPY_data_loc)

        SPY_data['datetime_bins'] = pd.to_datetime(SPY_data['datetime_bins'])
        SPY_data.rename(columns={"datetime_bins": "datetime_bins_prev"}, inplace=True)
        SPY_data.set_index("datetime_bins_prev", inplace=True)

        SPY_data['datetime_bins'] = SPY_data.index.ceil(delta)

        # Group by the datetime_bins and calculate the log returns
        SPY_returns = SPY_data.groupby("datetime_bins").apply(
            lambda x: pd.Series({
                'log_ret': np.log(x['last_midprice'].iloc[-1]) - np.log(x['first_midprice'].iloc[0])
            })
        ).reset_index()

        grouped['log_ret_ex'] = grouped['log_ret'] - SPY_returns['log_ret']
        grouped['fut_log_ret_ex'] = grouped['log_ret_ex'].shift(-1)
        grouped['date'] = pd.to_datetime(current_date)

        ff_df = pd.read_csv("/nfs/home/jingt/dissertation-iceberg/fama_french/F-F_Research_Data_5_Factors_2x3_daily.CSV",
        skiprows=[0, 1, 2])
        ff_df = ff_df.iloc[:-1, :]

        ff_df['Unnamed: 0'] = pd.to_datetime(ff_df['Unnamed: 0'], format='%Y%m%d')

        # Set this column as the index
        ff_df.set_index('Unnamed: 0', inplace=True)
        ff_df.index.name = 'datetime'
        # Filter the dataframe to keep only the dates beyond 2017
        ff_df = ff_df[ff_df.index >= '2018-01-01']


        grouped = pd.merge(grouped, ff_df[['SMB', 'HML', 'RF', 'CMA', 'RMW']], left_on="date", right_index=True, how='left')
        grouped.drop(columns=['date'], inplace=True)



    ### This commented version uses files with fret calculations_______________________
    # if delta == 'daily':
    #     ff_df = pd.read_csv("/nfs/home/jingt/dissertation-iceberg/fama_french/F-F_Research_Data_Factors_daily.CSV",
    #                 skiprows=5)
    #     ff_df = ff_df.iloc[:-1, :]

    #     ff_df['Unnamed: 0'] = pd.to_datetime(ff_df['Unnamed: 0'], format='%Y%m%d')

    #     # Set this column as the index
    #     ff_df.set_index('Unnamed: 0', inplace=True)
    #     ff_df.index.name = 'datetime'
    #     # Filter the dataframe to keep only the dates beyond 2017
    #     ff_df = ff_df[ff_df.index >= '2018-01-01']

    #     date_str = current_date.strftime('%Y-%m-%d')
    #     fret_folder = '/nfs/home/jingt/dissertation-iceberg/data/fret_folder'
    #     file_name = f'{date_str}.csv.gz'
    #     file_path = os.path.join(fret_folder, file_name)

    #     if os.path.exists(file_path):
    #         # Open and read the CSV file inside the .gz
    #         with gzip.open(file_path, 'rt') as f:
    #             df = pd.read_csv(f)
    #         print(f"File for {date_str} opened successfully.")

    #         if not df[df['Ticker'] == ticker]['fret_CLOP_MR'].empty and not df[df['Ticker'] == ticker]['fret_CLCL_MR'].empty:
    #             grouped['fret_ClOp'] = df[df['Ticker'] == ticker]['fret_CLOP_MR'].iloc[0]
    #             grouped['fret_ClOp'] -= (df[df['Ticker'] == 'SPY']['fret_CLOP_MR'].iloc[0])
    #             grouped['fret_ClCl'] = df[df['Ticker'] == ticker]['fret_CLCL_MR'].iloc[0]
    #             grouped['fret_ClCl'] -= (df[df['Ticker'] == 'SPY']['fret_CLCL_MR'].iloc[0])

    #         else:
    #             data = fetch_ticker_data_with_retry(ticker, current_date, retries=10, delay=5)
    #             dataSPY = fetch_ticker_data_with_retry('SPY', current_date, retries=10, delay=5)

    #             grouped['fret_ClOp'] = np.log(data['Open'].iloc[1]/ data['Close'].iloc[0])
    #             grouped['fret_ClOp'] -= (np.log(dataSPY['Open'].iloc[1] / dataSPY['Close'].iloc[0]))

    #             grouped['fret_ClCl'] = np.log(data['Close'].iloc[1] / data['Close'].iloc[0])
    #             grouped['fret_ClCl'] -= (np.log(dataSPY['Close'].iloc[1] / dataSPY['Close'].iloc[0])) 



    #     else:
    #         print(f"No file found for date {date_str} at {file_path}.")
    #         data = fetch_ticker_data_with_retry(ticker, current_date, retries=10, delay=5)
    #         dataSPY = fetch_ticker_data_with_retry('SPY', current_date, retries=10, delay=5)

    #         grouped['fret_ClOp'] = np.log(data['Open'].iloc[1] / data['Close'].iloc[0])
    #         grouped['fret_ClOp'] -= (np.log(dataSPY['Open'].iloc[1] / dataSPY['Close'].iloc[0]))
            
    #         grouped['fret_ClCl'] = np.log(data['Close'].iloc[1] / data['Close'].iloc[0])
    #         grouped['fret_ClCl'] -= (np.log(dataSPY['Close'].iloc[1] / dataSPY['Close'].iloc[0]))

    #_______________________

    # Calculate returns for daily timeframe
    # Version below directly uses yfinance
    if delta == 'daily':

        # Load Fama-French factors
        ff_df = pd.read_csv("/nfs/home/jingt/dissertation-iceberg/fama_french/F-F_Research_Data_5_Factors_2x3_daily.CSV",
        skiprows=[0, 1, 2])
        ff_df = ff_df.iloc[:-1, :]

        ff_df['Unnamed: 0'] = pd.to_datetime(ff_df['Unnamed: 0'], format='%Y%m%d')

        ff_df.set_index('Unnamed: 0', inplace=True)
        ff_df.index.name = 'datetime'
        ff_df = ff_df[ff_df.index >= '2018-01-01']

        # Get data for ticker and SPY around current trading day
        data = fetch_ticker_data_with_retry(ticker, current_date, retries=10, delay=5, include_prev=True)

        if data.index.tz is not None:
            current_date = pd.to_datetime(current_date).tz_localize(data.index.tz)

        next_date = data.index[data.index > current_date][0]
        prev_date = data.index[data.index < current_date][-1]

        # Calculate returns
        grouped['fret_ClOp'] = np.log(data['Open'].loc[next_date, ticker] / data['Close'].loc[current_date, ticker])
        grouped['fret_ClOp'] -= np.log(data['Open'].loc[next_date, 'SPY'] / data['Close'].loc[current_date, 'SPY'])

        grouped['ClOp'] = np.log(data['Open'].loc[current_date, ticker] / data['Close'].loc[prev_date, ticker])
        grouped['ClOp'] -= np.log(data['Open'].loc[current_date, 'SPY'] / data['Close'].loc[prev_date, 'SPY'])

        grouped = pd.merge(grouped, ff_df[['SMB', 'HML', 'RF', 'CMA', 'RMW']], left_on='datetime_bins', right_index=True, how='left')
        
            
        grouped['fret_adjClOp'] = np.log(data['Open'].loc[next_date, ticker] / data['Adj Close'].loc[current_date, ticker])
        grouped['fret_adjClOp'] -= np.log(data['Open'].loc[next_date, 'SPY'] / data['Adj Close'].loc[current_date, 'SPY'])

        pd.set_option("display.max_columns", None)


    return grouped


def filter_quantiles(grouped, column, lower_quantile=0.025, upper_quantile=0.975):
    # Filter quantiles for log returns
    lower_bound = grouped[column].quantile(lower_quantile)
    upper_bound = grouped[column].quantile(upper_quantile)
    return grouped[(grouped[column] >= lower_bound) & (grouped[column] <= upper_bound)]




def calculate_order_imbalance(df, direction_column, size_column, pred_dir_column=None):
    # Calculate weighted or original order imbalance 
    if pred_dir_column:
        return (df[size_column] * (1 - 2 * df[pred_dir_column])).sum() / df[size_column].sum()
    else:
        buy_size = df.loc[df[direction_column] == -1, size_column].sum()
        sell_size = df.loc[df[direction_column] == 1, size_column].sum()
        return (buy_size - sell_size) / (sell_size + buy_size)





def order_imbalance(df_full, df_pred=None, df_ob=None, delta='30S', type='vis'):
    # Calculate order imbalance for vis, hid or all orders
    weight = df_ob['bid_size_1'] / (df_ob['bid_size_1'] + df_ob['ask_size_1'])
    df_full['weighted_mp'] = weight * df_ob['ask_price_1'] + (1 - weight) * df_ob['bid_price_1']

    if type == 'vis':
        df = df_full[df_full['event_type'] == 4]
    elif type == 'hid':
        df = df_full[df_full['event_type'] == 5]
    elif type == 'all':
        df = df_full[df_full['event_type'].isin([4, 5])]
    else:
        print("Not Implemented")
        return
    if df.empty:
        return df

    if delta == 'daily':
        df_full['datetime_bins'] = df_full.index.get_level_values('datetime').normalize()
        df['datetime_bins'] = df.index.get_level_values('datetime').normalize()

    else:
        df_full['datetime_bins'] = df_full.index.get_level_values('datetime').ceil(delta)
        df['datetime_bins'] = df.index.get_level_values('datetime').ceil(delta)


    # Create the DataFrame with the datetime bins
    full_bins = get_datetime_bins(df_full, delta)

    if type == 'hid':
        df = pd.merge(df.reset_index(), df_pred.reset_index(), on=['datetime', 'ticker', 'event_number']).set_index(['datetime', 'ticker', 'event_number'])

    if type == 'all':
        df = pd.merge(df.reset_index(), df_pred.reset_index(), on=['datetime', 'ticker', 'event_number']).set_index(['datetime', 'ticker', 'event_number'], how='left')
        # If all, configure pred prob column accordingly for weighted OI
        df.loc[(df['event_type'] == 4) & (df['direction'] == 1), 'pred_prob'] = 1
        df.loc[(df['event_type'] == 4) & (df['direction'] == -1), 'pred_prob'] = 0

    # Calculate order imbalance for dataframe
    grouped = df.groupby('datetime_bins').apply(
        lambda x: pd.Series({
            'order_imbalance': calculate_order_imbalance(x, 'direction', 'size', 'pred_prob' if (type == 'hid' or type == 'all') else None),
        })
    ).reset_index()

    # Merge grouped data with full bins to include all datetime bins
    grouped = pd.merge(full_bins, grouped, on='datetime_bins', how='left').fillna(0)

    # Calculate first and last midprice and weighted_mp based on df_full
    full_grouped = df_full.groupby('datetime_bins').agg({
        'midprice': ['first', 'last'],
        'weighted_mp': ['first', 'last']
    }).reset_index()

    full_grouped.columns = ['datetime_bins', 'first_midprice', 'last_midprice', 'first_weighted_mp', 'last_weighted_mp']

    # Merge with grouped to include these values
    grouped = pd.merge(grouped, full_grouped, on='datetime_bins', how='left')

    ticker = df_full.index.get_level_values('ticker')[0]

    grouped['order_imbalance'] = grouped['order_imbalance'].fillna(0)

    # Include all other forms of returns
    grouped = calculate_log_returns(grouped, delta=delta, ticker=ticker)

    if delta == "daily":
        return grouped
    else:
        return grouped[:-1]



def combined_order_imbalance(df_full, df_pred, df_ob, delta='5min'):
    # Combine lit and hidden COI for combined signals
    df_vis = order_imbalance(df_full=df_full, df_ob=df_ob, delta=delta, type='vis')
    df_hid = order_imbalance(df_full=df_full, df_ob=df_ob, df_pred=df_pred, delta=delta, type='hid')


    df_comb = df_vis.merge(df_hid[['datetime_bins', 'order_imbalance']], on='datetime_bins', suffixes=('_vis', '_hid'))
    return df_comb


def get_datetime_bins(df_full, delta):
    # Get datetime bins based on timeframe
    if delta != 'daily':
        # Extract the first and last date from the datetime column
        start_date = df_full.index.get_level_values('datetime').min().date()
        end_date = df_full.index.get_level_values('datetime').max().date()

        # Set the time to start at 09:30 and end at 15:30
        start_datetime = pd.Timestamp.combine(start_date, pd.Timestamp("10:00").time())
        start_datetime = start_datetime + pd.Timedelta(delta)
        end_datetime = pd.Timestamp.combine(end_date, pd.Timestamp("15:30").time())

        # Generate the full range of datetime bins with the specified frequency
        full_range = pd.date_range(start=start_datetime, end=end_datetime, freq=delta)

        # Create the DataFrame with the datetime bins
        full_bins = pd.DataFrame(full_range, columns=['datetime_bins']) 

    else:
        full_range = df_full.index.get_level_values('datetime').normalize()[0]
        full_bins = pd.DataFrame([full_range], columns=['datetime_bins'])
    return full_bins


def iceberg_order_imbalance(df_full, df_pred, df_ob, delta='5min', weighted=False):
    # Calculate VHI order imbalance
    weight = df_ob['bid_size_1'] / (df_ob['bid_size_1'] + df_ob['ask_size_1'])
    df_full['weighted_mp'] = weight * df_ob['ask_price_1'] + (1 - weight) * df_ob['bid_price_1']
    ib_delta = '1ms'
    df = iceberg_tag(df_full, ib_delta)

    full_bins = get_datetime_bins(df_full, delta)

    if delta == 'daily':
        df_full['datetime_bins'] = df_full.index.get_level_values('datetime').normalize()
        df['datetime_bins'] = df.index.get_level_values('datetime').normalize()

    else:
        df_full['datetime_bins'] = df_full.index.get_level_values('datetime').ceil(delta)
        df['datetime_bins'] = df.index.get_level_values('datetime').ceil(delta)



    grouped = df.groupby('datetime_bins').apply(
        lambda x: pd.Series({
            'order_imbalance_vis': calculate_order_imbalance(x[x['iceberg'] == 0], 'direction', 'size'),
            'order_imbalance_ib': calculate_order_imbalance(x[x['iceberg'] == 1], 'direction', 'size'),
        })
    ).reset_index()

    # Merge grouped data with full bins to include all bins
    grouped = pd.merge(full_bins, grouped, on='datetime_bins', how='left').fillna(0)

    # Calculate first and last midprice and weighted_mp based on df_full
    full_grouped = df_full.groupby('datetime_bins').agg({
        'midprice': ['first', 'last'],
        'weighted_mp': ['first', 'last']
    }).reset_index()

    full_grouped.columns = ['datetime_bins', 'first_midprice', 'last_midprice', 'first_weighted_mp', 'last_weighted_mp']

    grouped['order_imbalance_vis'] = grouped['order_imbalance_vis'].fillna(0)
    grouped['order_imbalance_ib'] = grouped['order_imbalance_ib'].fillna(0)

    # Merge with grouped to include these values
    grouped = pd.merge(grouped, full_grouped, on='datetime_bins', how='left')

    ticker = df_full.index.get_level_values('ticker')[0]
    grouped = calculate_log_returns(grouped, delta=delta, ticker=ticker)

    df_hid = order_imbalance(df_full=df_full, df_pred=df_pred, df_ob=df_ob, delta=delta, type='hid')
    df_hid.rename(columns={"order_imbalance": "order_imbalance_hid"}, inplace=True)

    grouped = pd.merge(grouped, df_hid[['datetime_bins', 'order_imbalance_hid']], on='datetime_bins', how='left')

    grouped['order_imbalance_hid'] = grouped['order_imbalance_hid'].fillna(0)

    return grouped



def agg_order(df, df_pred, agg='agg_low'):
    # Calculate conditional OI based on aggressiveness of orders
    df = df[df['event_type']==5]

    if agg=='agg_low':
        return df[((df_pred['pred_dir'] == 1) & (df['agg_ratio'] < 0.5)) | 
                  ((df_pred['pred_dir'] == -1) & (df['agg_ratio'] > 0.5))]
    
    elif agg=='agg_mid':
        return df[df['agg_ratio'] == 0.5]

    elif agg=='agg_high':
        return df[((df_pred['pred_dir'] == 1) & (df['agg_ratio'] > 0.5)) | 
                  ((df_pred['pred_dir'] == -1) & (df['agg_ratio'] < 0.5))]


def size_order(df, df_pred, size='small'):
    # Calculate conditional OI based on size of orders
    df = df[df['event_type']==5]
    lower_q = df['size'].quantile(1/3)
    upper_q = df['size'].quantile(2/3)

    if size=='small':
        return df[df['size'] < lower_q]
    
    elif size=='medium':
        return df[(df['size'] >= lower_q) & (df['size'] <= upper_q)]

    elif size=='large':
        return df[df['size'] > upper_q]



def conditional_order_imbalance(df_full, df_pred, df_ob, delta='5min', condition='agg'):
    # Calculate conditional OI based on either aggressiveness or size
    df_fin = order_imbalance(df_full=df_full, delta=delta, df_ob=df_ob, type='vis')
    df_fin.rename(columns={'order_imbalance': 'order_imbalance_vis'}, inplace=True)

    if condition == 'agg':
        for version, suffix in zip(['agg_low', 'agg_mid', 'agg_high'], ['_agg_low', '_agg_mid', '_agg_high']):
            df = order_imbalance(df_full=agg_order(df_full, df_pred, agg=version), df_pred=df_pred, df_ob=df_ob, delta=delta, type='hid')
            if not df.empty:
                df.rename(columns={'order_imbalance': f'order_imbalance{suffix}'}, inplace=True)
                df_fin = df_fin.merge(df[['datetime_bins', f'order_imbalance{suffix}']], on='datetime_bins')
            else:
                df_fin[f'order_imbalance{suffix}'] = 0

    elif condition == 'size':
        for version in ['small', 'medium', 'large']:
            df = order_imbalance(df_full=size_order(df_full, df_pred, size=version), df_pred=df_pred, df_ob=df_ob, delta=delta, type='hid')
            if not df.empty:
                df.rename(columns={'order_imbalance': f'order_imbalance_{version}'}, inplace=True)
                df_fin = df_fin.merge(df[['datetime_bins', f'order_imbalance_{version}']], on='datetime_bins')
            else:
                df_fin[f'order_imbalance_{version}'] = 0

    return df_fin



# def lm_results(df_full, df_pred, df_ob, delta_lst, order_type='combined', predictive=True, ret_type='log_ret',
#                momentum=False):
    
#     if ret_type == 'log_ret':
#         y = "fut_log_ret" if predictive else "log_ret"
#         x_momentum = "+ log_ret" if momentum else ""

#     elif ret_type == 'weighted_mp':
#         y = 'fut_weighted_log_ret' if predictive else "weighted_log_ret"
#         x_momentum = "+ weighted_log_ret" if momentum else ""

#     elif ret_type == 'tClose':
#         y = "fret_tClose" if predictive else "ret_tClose"
#         x_momentum = "+ ret_tClose" if momentum else ""
    
#     elif ret_type == 'ClOp' or ret_type == 'ClCl' or ret_type == 'adjClOp':
#         y = f"fret_{ret_type}"

#     elif ret_type == 'daily_ret':
#         y = "fut_daily_ret"

    
#     params_lst = []
#     tvalues_lst = []

#     for delta in delta_lst:
#         if order_type == 'vis' or order_type == 'hid' or order_type == 'combined' or order_type == 'all':
#             df_merged = combined_order_imbalance(df_full, df_pred, df_ob, delta=delta)

#         elif order_type == 'comb_iceberg':
#             df_merged = iceberg_order_imbalance(df_full, df_pred, df_ob, delta=delta)
#             lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis + order_imbalance_hid + order_imbalance_ib + {x_momentum}""", 
#                          data=df_merged).fit()
#             params_lst.append((lm.params[1], lm.params[2], lm.params[3]))
#             tvalues_lst.append((lm.tvalues[1], lm.tvalues[2], lm.tvalues[3]))

#         elif order_type == 'agg':
#             df_merged = conditional_order_imbalance(df_full, df_pred, df_ob, delta=delta, condition='agg')
#             lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis + order_imbalance_agg_low + \n
#                          order_imbalance_agg_mid + order_imbalance_agg_high {x_momentum}""", 
#                          data=df_merged).fit()
#             params_lst.append((lm.params[1], lm.params[2], lm.params[3], lm.params[4]))
#             tvalues_lst.append((lm.tvalues[1], lm.tvalues[2], lm.tvalues[3], lm.tvalues[4]))
        
#         elif order_type == 'size':
#             df_merged = conditional_order_imbalance(df_full, df_pred, df_ob, delta=delta, condition='size')
#             lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis + order_imbalance_small + \n
#                          order_imbalance_medium + order_imbalance_large {x_momentum}""", 
#                          data=df_merged).fit()
#             params_lst.append((lm.params[1], lm.params[2], lm.params[3], lm.params[4]))
#             tvalues_lst.append((lm.tvalues[1], lm.tvalues[2], lm.tvalues[3], lm.tvalues[4]))

#         if order_type == "vis":
#             lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis {x_momentum}""", data=df_merged).fit()
#             params_lst.append(lm.params[1])
#             tvalues_lst.append(lm.tvalues[1])

#         elif order_type == "hid":
#             lm = smf.ols(formula=f"""{y} ~ order_imbalance_hid {x_momentum}""", data=df_merged).fit()
#             params_lst.append(lm.params[1])
#             tvalues_lst.append(lm.tvalues[1])

#         elif order_type == "combined":
#             lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis + order_imbalance_hid {x_momentum}""", data=df_merged).fit()
#             params_lst.append((lm.params[1], lm.params[2]))
#             tvalues_lst.append((lm.tvalues[1], lm.tvalues[2]))


#     if order_type == "vis" or order_type == "hid":
#         return pd.DataFrame({"timeframe": delta_lst, "params": params_lst, "tvalues": tvalues_lst})

#     elif order_type == "combined":
#         df = pd.DataFrame({
#             'timeframe': delta_lst,
#             'params_vis': [x[0] for x in params_lst],
#             'tvalues_vis': [x[0] for x in tvalues_lst],
#             'params_hid': [x[1] for x in params_lst],
#             'tvalues_hid': [x[1] for x in tvalues_lst]
#         })
#         return df


#     elif order_type == 'comb_iceberg':
#         df = pd.DataFrame({
#             'timeframe': delta_lst,
#             'params_vis': [x[0] for x in params_lst],
#             'tvalues_vis': [x[0] for x in tvalues_lst],
#             'params_hid': [x[1] for x in params_lst],
#             'tvalues_hid': [x[1] for x in tvalues_lst],
#             'params_ib': [x[2] for x in params_lst],
#             'tvalues_ib': [x[2] for x in tvalues_lst]
#         })
#         return df


#     elif order_type == 'agg':
#         df = pd.DataFrame({
#             'timeframe': delta_lst,
#             'params_vis': [x[0] for x in params_lst],
#             'tvalues_vis': [x[0] for x in tvalues_lst],
#             'params_agg_low': [x[1] for x in params_lst],
#             'tvalues_agg_low': [x[1] for x in tvalues_lst],
#             'params_agg_mid': [x[2] for x in params_lst],
#             'tvalues_agg_mid': [x[2] for x in tvalues_lst],
#             'params_agg_high': [x[3] for x in params_lst],
#             'tvalues_agg_high': [x[3] for x in tvalues_lst],
#         })
#         return df
    
#     elif order_type == 'size':
#         df = pd.DataFrame({
#             'timeframe': delta_lst,
#             'params_vis': [x[0] for x in params_lst],
#             'tvalues_vis': [x[0] for x in tvalues_lst],
#             'params_small': [x[1] for x in params_lst],
#             'tvalues_small': [x[1] for x in tvalues_lst],
#             'params_medium': [x[2] for x in params_lst],
#             'tvalues_medium': [x[2] for x in tvalues_lst],
#             'params_large': [x[3] for x in params_lst],
#             'tvalues_large': [x[3] for x in tvalues_lst],
#         })
#         return df


def diagnostic_plots(lm, y, data):
    # fitted values (need a constant term for intercept)
    model_fitted_y = lm.fittedvalues

    # model residuals
    model_residuals = lm.resid

    # normalized residuals
    model_norm_residuals = lm.get_influence().resid_studentized_internal

    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

    # absolute residuals
    model_abs_resid = np.abs(model_residuals)

    # leverage
    model_leverage = lm.get_influence().hat_matrix_diag

    # cook's distance
    model_cooks = lm.get_influence().cooks_distance[0]
    
    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Fitted
    sns.residplot(x=model_fitted_y, y=y, data=data, 
                  lowess=True, 
                  scatter_kws={'alpha': 0.5}, 
                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                  ax=axs[0, 0])
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')

    # Normal Q-Q
    QQ = ProbPlot(model_norm_residuals)
    QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1, ax=axs[0, 1])
    axs[0, 1].set_title('Normal Q-Q')
    axs[0, 1].set_xlabel('Theoretical Quantiles')
    axs[0, 1].set_ylabel('Standardized Residuals')

    # Scale-Location
    axs[1, 0].scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(x=model_fitted_y, y=model_norm_residuals_abs_sqrt, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                ax=axs[1, 0])
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel('$\sqrt{|Standardized Residuals|}$')

    # Residuals vs Leverage
    axs[1, 1].scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(x=model_leverage, y=model_norm_residuals, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                ax=axs[1, 1])
    axs[1, 1].set_title('Residuals vs Leverage')
    axs[1, 1].set_xlabel('Leverage')
    axs[1, 1].set_ylabel('Standardized Residuals')

    plt.tight_layout()
    plt.show()
