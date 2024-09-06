import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
import py7zr

from prediction_ML_pipeline import extract_info_from_filename, add_date_ticker, data_preprocessing, save_dataframe_to_folder
from order_imbalance import iceberg_tag



def mean_analysis(ticker, calc_type='mean', year=2018, delta='10min', iceberg=False):
    # Get individual dataframes of different COI for a ticker
    df_4_lst = []
    df_5_lst = []
    df_4_ib_lst = []


    archive_path = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{year}-01-01_{year}-12-31_10.7z"
    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        filenames = archive.getnames()
        orderbook_files = [f for f in filenames if 'orderbook' in f]
        message_files = [f for f in filenames if 'message' in f]
        # Process matching orderbook and message files
        for orderbook_file, message_file in zip(orderbook_files, message_files):
            extracted_files = archive.read([orderbook_file, message_file])
            orderbook_stream = io.BytesIO(extracted_files[orderbook_file].read())
            message_stream = io.BytesIO(extracted_files[message_file].read())
            print("Processed files:", orderbook_file, message_file, flush=True)

            orderbook_chunk = pd.read_csv(orderbook_stream, header=None, usecols=[0, 1, 2, 3])
            message_chunk = pd.read_csv(message_stream, header=None, usecols=[0, 1, 2, 3, 4, 5])
            ticker, date = extract_info_from_filename(message_file)

            message_chunk = add_date_ticker(message_chunk, date, ticker)

            message_chunk, orderbook_chunk = data_preprocessing(message_chunk, orderbook_chunk, ticker_name=ticker,
                                                                start_time="9:30:00", end_time='4:00:00')
            
            df_4 = iceberg_tag(message_chunk, ib_delta='1ms')
            df_5 = message_chunk[message_chunk['event_type'] == 5]

            df_4_lst.append(df_4[df_4['iceberg']==0])
            df_5_lst.append(df_5)
            df_4_ib_lst.append(df_4[df_4['iceberg']==1])

        df_4_full = pd.concat(df_4_lst)
        df_5_full = pd.concat(df_5_lst)
        df_ib_full = pd.concat(df_4_ib_lst)


        return df_4_full, df_5_full, df_ib_full
        

def process_df(df):
    df = df.to_frame()
    df.index = pd.to_datetime(df.index)
    df['time'] = df.index.time
    df['month'] = df.index.month
    df['date'] = df.index.date

    return df


def summary_stat(ticker_lst, calc_type='mean', year=2018, delta='10min', iceberg=True):
    # Get summary statistic of COI dfs for each ticker
    summary_stat_4 = []
    summary_stat_5 = []
    summary_stat_ib = []


    for ticker in ticker_lst:
        print("For ticker: {ticker}", flush=True)
        df_4_full, df_5_full, df_ib_full = mean_analysis(ticker, calc_type=calc_type,
                                                         year=year, delta=delta, iceberg=True)
        # Describe the 'size' column
        size_desc_4 = df_4_full['size'].describe()
        size_desc_5 = df_5_full['size'].describe()
        size_desc_ib = df_ib_full['size'].describe()
        
        # Convert the Series to a dictionary and add the ticker to it
        size_desc_dict_4 = size_desc_4.to_dict()
        size_desc_dict_4['Ticker'] = ticker

        size_desc_dict_5 = size_desc_5.to_dict()
        size_desc_dict_5['Ticker'] = ticker

        size_desc_dict_ib = size_desc_ib.to_dict()
        size_desc_dict_ib['Ticker'] = ticker

        print(size_desc_dict_4, flush=True)
        print(size_desc_dict_5, flush=True)
        print(size_desc_dict_ib, flush=True)

        
        # Append the dictionary to the list
        summary_stat_4.append(size_desc_dict_4)
        summary_stat_5.append(size_desc_dict_5)
        summary_stat_ib.append(size_desc_dict_ib)
        print("Done!", flush=True)


    # Convert list of dicts to a dataframe
    summary_df_4 = pd.DataFrame(summary_stat_4)
    summary_df_5 = pd.DataFrame(summary_stat_5)
    summary_df_ib = pd.DataFrame(summary_stat_ib)

    # Move 'Ticker' column to the first position
    summary_df = summary_df[['Ticker'] + [col for col in summary_df.columns if col != 'Ticker']]

    folder_path = "/nfs/home/jingt/dissertation-iceberg/data/output_results"
    save_dataframe_to_folder(summary_df_4, folder_path, 'summary_4')
    save_dataframe_to_folder(summary_df_5, folder_path, 'summary_5')
    save_dataframe_to_folder(summary_df_ib, folder_path, 'summary_ib')


ticker_lst = ['AES', 'ALB', 'AOS', 'APA', 'BEN', 'BXP', 'CPB',
                'DVA', 'FFIV', 'FRT', 'HII', 'HRL', 'HSIC', 'INCY',
                'MHK', 'NWSA', 'PNW', 'RL', 'TAP', 'WYNN']


def main():
    summary_stat(ticker_lst, calc_type='agg', year=2018, delta='10min', iceberg=True)

if __name__ == '__main__':
    main()