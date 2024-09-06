import pandas as pd
import matplotlib.pyplot as plt
from chunk_pipeline import order_imbalance_calc
import warnings
warnings.filterwarnings("ignore")
from prediction_ML_pipeline import save_dataframe_to_folder

from statsmodels.graphics.tsaplots import plot_acf, acf
from statsmodels.graphics.tsaplots import plot_pacf, pacf

def main():

    ticker_lst = ['AES', 'ALB', 'AOS', 'APA', 'BEN', 'BXP', 'CPB',
                'DVA', 'FFIV', 'FRT', 'HII', 'HRL', 'HSIC', 'INCY',
                'MHK', 'NWSA', 'PNW', 'RL', 'TAP', 'WYNN']

    order_type = 'comb_iceberg'
    delta_lst = ['15S', '30S', '1min', '2min', '5min', '10min']

    df_lst = []

    for ticker in ticker_lst:

        folder_path = "/nfs/home/jingt/dissertation-iceberg/data/output_results"
        model_path = "/nfs/home/jingt/dissertation-iceberg/data/output_folder"
        archive_train_path = f"/nfs/home/jingt/dissertation-iceberg/data/training_data/_data_dwn_32_210_hidden_liquidity_{ticker}_2012-01-01_2012-12-31_1.7z"
        arhive_test_path = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_2018-01-01_2018-12-31_10.7z"
        model_name = f"xgboost_{ticker}.json"

        df_dict = order_imbalance_calc(arhive_test_path, delta_lst=delta_lst, model=None,
                                        model_path=model_path, model_name=model_name, 
                                        order_type=order_type)
        
        for delta in delta_lst:
            correlation_vis_hid = df_dict[delta]['order_imbalance_vis'].corr(df_dict[delta]['order_imbalance_hid'])
            correlation_vis_ib = df_dict[delta]['order_imbalance_vis'].corr(df_dict[delta]['order_imbalance_ib'])
            correlation_hid_ib = df_dict[delta]['order_imbalance_hid'].corr(df_dict[delta]['order_imbalance_ib'])


            df_ticker = pd.DataFrame({'Ticker': [ticker], 'delta': [delta], 'vishid': [correlation_vis_hid], 'visib': [correlation_vis_ib], 'hidib': [correlation_hid_ib]})

            df_lst.append(df_ticker)
        
        df_all = pd.concat(df_lst)
        print(df_all, flush=True)
        file_name = 'correlation_iceberg.csv'
        save_dataframe_to_folder(df_all, "/nfs/home/jingt/dissertation-iceberg/data/output_results/2018", file_name)


if __name__ == '__main__':
    main()


