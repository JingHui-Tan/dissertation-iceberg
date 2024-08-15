import py7zr
import pandas as pd
import io
import numpy as np

from prediction_ML_pipeline import data_preprocessing, prediction_feature, add_date_ticker, extract_info_from_filename, hid_outside_spread_tag, save_dataframe_to_folder
from order_imbalance import order_imbalance, combined_order_imbalance, conditional_order_imbalance, iceberg_order_imbalance


archive_path = "/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__SPY_2018-01-01_2018-12-31_10.7z"
folder_path = "/nfs/home/jingt/dissertation-iceberg/data/SPY_data"

def main():
    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        filenames = archive.getnames()
        orderbook_files = [f for f in filenames if 'orderbook' in f]
        message_files = [f for f in filenames if 'message' in f]

        for orderbook_file, message_file in zip(orderbook_files, message_files):
            extracted_files = archive.read([orderbook_file, message_file])
            orderbook_stream = io.BytesIO(extracted_files[orderbook_file].read())
            message_stream = io.BytesIO(extracted_files[message_file].read())
            print("Processed files:", orderbook_file, message_file)

            # Read the entire CSV files
            orderbook_df = pd.read_csv(orderbook_stream, header=None, usecols=[0, 1, 2, 3])
            message_df = pd.read_csv(message_stream, header=None, usecols=[0, 1, 2, 3, 4, 5])

            ticker, date = extract_info_from_filename(message_file)

            # Process data for prediction
            message_df = add_date_ticker(message_df, date, ticker)
            message_df, orderbook_df = data_preprocessing(message_df, orderbook_df, ticker_name=ticker)

            message_df['midprice'] = (orderbook_df['ask_price_1'] + orderbook_df['bid_price_1']) / 2
            message_df['datetime'] = message_df.index.get_level_values('datetime')
            message_df = message_df[['datetime', 'midprice']]

            message_df['datetime'] = pd.to_datetime(message_df['datetime'])
            message_df.set_index('datetime', inplace=True)
            message_df['datetime_bins'] = message_df.index.ceil('15S')

            grouped = message_df.groupby('datetime_bins').apply(
                lambda x: pd.Series({
                'first_midprice': x['midprice'].iloc[0],
                'last_midprice': x['midprice'].iloc[-1],
                })).reset_index()

            file_name = f"SPY_{date}.csv"
            print(f"Saving {file_name}", flush=True)
            save_dataframe_to_folder(grouped, folder_path, file_name)


if __name__ == "__main__":
    main()