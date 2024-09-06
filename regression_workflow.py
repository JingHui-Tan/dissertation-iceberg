from chunk_pipeline import *
from prediction_ML_pipeline import save_dataframe_to_folder
import time
import os

def main():
    logging.getLogger().setLevel(logging.WARNING)

    ## ---------CONFIGURE THIS FOR REQUIRED COMPUTATION---------------

    ticker_lst = ['AES', 'ALB', 'AOS', 'APA', 'BEN', 'BXP', 'CPB',
                  'DVA', 'FFIV', 'FRT', 'HII', 'HRL', 'HSIC', 'INCY',
                  'MHK', 'NWSA', 'PNW', 'RL', 'TAP', 'WYNN']

    
    skip_training = True
    year = "2018"
    order_type_lst = ['all']
    predictive_lst = [True]
    ret_type_lst = ["ClOp"]
    momentum_lst = [True]
    file_name_lst= ["pred_ClOp_all.csv"]


    # delta_lst = ['30S', '1min', '2min', '5min', '10min', '15min', '30min']
    delta_lst = ['daily']

    ## ------------------------------------


    # Check if files exist
    full_results_df = None
    folder_path = "/nfs/home/jingt/dissertation-iceberg/data/output_regression"
    model_path = "/nfs/home/jingt/dissertation-iceberg/data/output_folder"
    results_df_all = {i: [] for i in file_name_lst}

    for ticker in ticker_lst:

        file_path_train = f'/nfs/home/jingt/dissertation-iceberg/data/training_data/_data_dwn_32_210_hidden_liquidity_{ticker}_2012-01-01_2012-12-31_1.7z'
        file_path_test = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{year}-01-01_{year}-12-31_10.7z"

        if not os.path.exists(file_path_train):
            print(f"The file {file_path_train} does not exist.")
        if not os.path.exists(file_path_test):
            print(f"The file {file_path_test} does not exist.")
    print("All files exist!")


    for ticker in ticker_lst:

        model_name = f"xgboost_{ticker}.json"

        print(f"For ticker: {ticker}")
        start = time.time()
        archive_train_path = f"/nfs/home/jingt/dissertation-iceberg/data/training_data/_data_dwn_32_210_hidden_liquidity_{ticker}_2012-01-01_2012-12-31_1.7z"
        arhive_test_path = f"/nfs/data/lobster_data/lobster_raw/2017-19/_data_dwn_32_302__{ticker}_{year}-01-01_{year}-12-31_10.7z"


        params = {
            'objective': 'binary:logistic', 
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 100,
            'eval_metric': 'logloss',  # Log loss for binary classification
        }

        if not skip_training:
            print("Model fitting started", flush=True)
            model, acc = process_and_train_xgb(archive_train_path, model_path, model_name, params=params)
            print("Model fitting completed!", flush=True)
        else:
            model = None
            acc = 0

        for order_type, file_name, predictive, ret_type, momentum in zip(order_type_lst, file_name_lst,
                                                                            predictive_lst, ret_type_lst,
                                                                            momentum_lst):
            print(f"For {order_type}, pred: {predictive}, ret_type: {ret_type}, momentum: {momentum}")
            print(f"{time.time() - start:.3f} seconds elapsed", flush=True)
            start = time.time()
            print("Order imbalance calculation started", flush=True)
            df_dict = order_imbalance_calc(arhive_test_path, delta_lst=delta_lst, model=model,
                                        model_path=model_path, model_name=model_name, 
                                        order_type=order_type)

            print("Order imbalance calculation completed!", flush=True)

            print(f"{time.time() - start:.3f} seconds elapsed", flush=True)

            start = time.time()

            print("Results construction started", flush=True)
            results_df = OI_results(df_dict, order_type=order_type, predictive=predictive,
                                    ret_type=ret_type, momentum=momentum)
            results_df['ticker'] = ticker
            results_df['model_acc'] = acc

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(results_df, flush=True)

            print("Results construction completed!", flush=True)

            print(f"{time.time() - start:.3f} seconds elapsed", flush=True)

            start = time.time()

            print("Saving to folder", flush=True)

            results_df_all[file_name].append(results_df)
            full_results_df = pd.concat(results_df_all[file_name])

            save_dataframe_to_folder(full_results_df, folder_path=folder_path, file_name=file_name)
            print("Saved to folder!", flush=True)


if __name__ == "__main__":
    main()