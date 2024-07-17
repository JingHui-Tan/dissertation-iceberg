
'''Conduct data preprocessing and ML directional prediction'''

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings


warnings.filterwarnings('ignore')

# class DataPreprocessor(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass

#     def fit(self, X, y=None, ticker_name=None):
#         self.ticker_name = ticker_name
#         return self

#     def transform(self, X, y=None):
#         df_m, df_ob = X
#         df_m_mh, df_ob_mh = data_preprocessing(df_m, df_ob, self.ticker_name)
#         return df_m_mh, df_ob_mh

# class FeatureCreator(BaseEstimator, TransformerMixin):
#     def __init__(self, labelled=False):
#         self.labelled = labelled

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         df_m_mh, df_ob_mh = X
#         if self.labelled:
#             features_df, output_df = prediction_feature(df_m_mh, df_ob_mh, labelled=self.labelled)
#             return {"X": features_df, "y": output_df}
#         else:
#             features_df = prediction_feature(df_m_mh, df_ob_mh, labelled=self.labelled)
#             return features_df
        


def data_preprocessing(df_m, df_ob, ticker_name=None):
    '''Preprocess message and orderbook dataframe'''

    df_m['ticker'] = ticker_name
    df_ob['ticker'] = ticker_name

    # Set up header for message df and OB df
    M_header = ['time', 'event_type', 'order_ID', 'size', 'price', 'direction', 'ticker']
    df_m = df_m.dropna(axis=1, how='all')
    df_m.columns = M_header

    OB_header = []
    for i in range(1, df_ob.shape[1]//4 + 1):
        OB_header.append(f'ask_price_{i}')
        OB_header.append(f'ask_size_{i}')
        OB_header.append(f'bid_price_{i}')
        OB_header.append(f'bid_size_{i}')

    OB_header.append('ticker')

    df_ob.columns = OB_header

    df_m['time'] = pd.to_timedelta(df_m['time'], unit='s')

    # Define the base date
    base_date = pd.Timestamp('2012-06-21')

    # Add the timedelta (time_sec) to the base date
    df_m['datetime'] = base_date + df_m['time']
    df_m.drop(columns=['time'], inplace=True)

    # Creating event number
    df_m['event_number'] = df_m.groupby(['datetime', 'ticker']).cumcount()
    df_m['event_number_at_t'] = df_m.groupby(['datetime', 'ticker'])['event_type'].transform('count')

    # Setting the composite index
    df_m.set_index(['datetime', 'ticker', 'event_number'], inplace=True)

    # Apply same index to OB df as message df
    df_ob.index = df_m.index

    # Define the start and end times
    start_time = pd.to_datetime("09:15:00").time()
    end_time = pd.to_datetime("15:45:00").time()

    # Extract the 'datetime' level from the MultiIndex and filter based on the time
    filtered_index = df_m.index.get_level_values('datetime').to_series().between_time(start_time, end_time).index

    # Use the filtered index to get the filtered DataFrame
    df_m_mh = df_m.loc[filtered_index]
    df_ob_mh = df_ob.loc[filtered_index]

    # Remove duplicates based on index
    df_m_mh = df_m_mh[~df_m_mh.index.duplicated(keep='first')]
    df_ob_mh = df_ob_mh[~df_ob_mh.index.duplicated(keep='first')]

    return df_m_mh, df_ob_mh







def direction_adjacent_event(df, event_type, order='prev'):
    """
    Get direction of previous or next event_type.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing event data.
    event_type (int): The event type to find direction for.
    order (str): Whether to find the 'prev' or 'next' event. Default is 'prev'.
    
    Returns:
    pd.DataFrame: Dataframe with added feature columns.
    """
    column_name = f"{order}_dir_t{event_type}"
    
    if order == 'prev':
        df[column_name] = np.where(df['event_type'] == event_type, df['direction'], np.nan)
        df[column_name] = df[column_name].shift(1).ffill().fillna(0)
    elif order == 'next':
        df[column_name] = np.where(df['event_type'] == event_type, df['direction'], np.nan)
        df[column_name] = df[column_name].shift(-1).bfill().fillna(0)
    
    return df


def trade_sentiment(df):
    '''Obtain trading sentiment before and after particular trade event'''
    conditions = [
    (df['event_type'] == 1),
    (df['event_type'] == 3),
    (df['event_type'] == 4)
    ]

    # Define corresponding values
    values = [
        df['direction'] * df['size'],
        -df['direction'] * df['size'],
        df['direction'] * df['size']
    ]

    # Apply np.select
    df['sentiment'] = np.select(conditions, values, default=0)

    df['agg_sentiment_prev'] = df['sentiment'].rolling(window=5).sum()
    df['agg_sentiment_prev'] = df['agg_sentiment_prev'].fillna(0)

    df['agg_sentiment_aft'] = df['sentiment'].iloc[::-1].rolling(window=5).sum().iloc[::-1]
    df['agg_sentiment_aft'] = df['agg_sentiment_aft'].iloc[::-1].fillna(0)

    df.drop(columns=['sentiment'], inplace=True)
    return df





def prediction_feature(df_m_mh, df_ob_mh, labelled=False, standardise=True):

    '''
    Create features for ML directional prediction.
    Parameters:
    df_m_mh (pd.DataFrame): The dataframe containing market and message data.
    df_ob_mh (pd.DataFrame): The dataframe containing order book data.
    
    Returns:
    features_hid (pd.DataFrame): Dataframe with added features.
    if labelled:
    Returns:
    features_hid (pd.DataFrame): Dataframe with added features.
    output_hid (pd.DataFrame): Dataframe of true direction
    '''

    # Get midprice at each timeframe
    df_m_mh['midprice'] = (df_ob_mh['ask_price_1'] + df_ob_mh['bid_price_1']) / 2

    # Compute Order Flow Imbalance (OFI)
    df_m_mh['ofi'] = (df_ob_mh['bid_size_1'] - df_ob_mh['ask_size_1']) / (df_ob_mh['bid_size_1'] + df_ob_mh['ask_size_1'])    

    # Get direction of specific events
    event_types = [1, 4, 3]
    for event_type in event_types:
        df_m_mh = direction_adjacent_event(df_m_mh, event_type=event_type, order='prev')
        df_m_mh = direction_adjacent_event(df_m_mh, event_type=event_type, order='next')

    df_m_mh = trade_sentiment(df_m_mh)

    df_m_mh['agg_ratio'] = (df_m_mh['price'] - df_ob_mh['bid_price_1']) / (df_ob_mh['ask_price_1'] - df_ob_mh['bid_price_1'])
    df_m_mh['bid_pref'] = df_ob_mh['bid_price_1'] / (df_ob_mh['bid_price_1'] + df_ob_mh['ask_price_1'])

    df_m_mh['hid_at_bid'] = (df_m_mh['price'] == df_ob_mh['bid_price_1']).astype(int) 
    df_m_mh['hid_at_ask'] = (df_m_mh['price'] == df_ob_mh['ask_price_1']).astype(int)

    # Extract event type 5 df and features, drop irrelevant features
    features_df = df_m_mh[df_m_mh['event_type'] == 5]
    features_df.drop(columns=['event_type', 'order_ID', 'price', 'direction', 'midprice'], inplace=True)
    numerical_columns = ['size', 'ofi', 'agg_ratio']

    # Standardise numerical columns
    if standardise:
        scaler = StandardScaler()
        features_df[numerical_columns] = scaler.fit_transform(features_df[numerical_columns])

    # Change type of binary columns to categorical
    categorical_columns = ['hid_at_bid', 'hid_at_ask', 'prev_dir_t1', 'next_dir_t1',
                           'prev_dir_t4', 'next_dir_t4', 'prev_dir_t3', 'next_dir_t3']
    

    for col in categorical_columns:
        features_df[col] = features_df[col].astype('category')


    # If labelled, then it is the training dataset used to fit the model
    if labelled:
        output_df = df_m_mh[df_m_mh['event_type']==5]['direction']
        return features_df, output_df # X_train, y_train
    
    return features_df # X_test






def hid_outside_spread_tag(pred_features_df, y_pred_df):
    '''Tag direction for hidden orders outside BA spread'''
    y_pred_df = y_pred_df.merge(pred_features_df['agg_ratio'], left_index=True, right_index=True)

    # Tag buy hidden liquidity execution
    y_pred_df.loc[y_pred_df['agg_ratio']<=0, "pred_dir"] = 1
    y_pred_df.loc[y_pred_df['agg_ratio']<=0, "pred_prob"] = 1

    # Tag sell hidden liquidity execution
    y_pred_df.loc[y_pred_df['agg_ratio']>=1, "pred_dir"] = -1
    y_pred_df.loc[y_pred_df['agg_ratio']>=1, "pred_prob"] = 0

    # Drop agg_ratio column
    y_pred_df.drop(columns=['agg_ratio'], inplace=True)

    return y_pred_df


def train_and_evaluate_model(classifier, grid, df_ob_labelled_lst, df_m_labelled_lst, tickers_train,
                             df_ob_predict_lst, df_m_predict_lst, tickers_pred):
    '''Perform model training and classification'''

    # Set up training data
    labelled_features_lst = []
    labelled_output_lst = []
    labelled_m_lst = []
    labelled_ob_lst = []

    for ticker, df_m, df_ob in zip(tickers_train, df_m_labelled_lst, df_ob_labelled_lst):
        df_m_mh, df_ob_mh = data_preprocessing(df_m, df_ob, ticker_name=ticker)
        features_hid, output_hid = prediction_feature(df_m_mh, df_ob_mh, labelled=True, standardise=True)
        # Either standardise here or within function - see which one works best

        labelled_features_lst.append(features_hid)
        labelled_output_lst.append(output_hid)
        labelled_m_lst.append(df_m_mh)
        labelled_ob_lst.append(df_ob_mh)
    
    labelled_features = pd.concat(labelled_features_lst)
    labelled_output = pd.concat(labelled_output_lst)

    X_train, X_test, y_train, y_test = train_test_split(labelled_features, labelled_output, test_size=0.25, random_state=42)

    # Fit ML model and do grid search
    grid_cv = GridSearchCV(classifier, param_grid=grid, cv=5,
                           scoring='accuracy', verbose=2)
    
    grid_cv.fit(X_train, y_train)

    # Print the best parameters and best accuracy from the grid search
    print("Best parameters found: ", grid_cv.best_params_)
    print("Best accuracy found: ", grid_cv.best_score_)

    # Use the best estimator directly
    best_classifier = grid_cv.best_estimator_

    # Set up predict data
    pred_features_lst = []
    pred_m_lst = []
    pred_ob_lst = []

    for ticker, df_m, df_ob in zip(tickers_pred, df_m_predict_lst, df_ob_predict_lst):
        df_m_mh, df_ob_mh = data_preprocessing(df_m, df_ob, ticker_name=ticker)
        features_hid = prediction_feature(df_m_mh, df_ob_mh, labelled=False, standardise=True)
        pred_features_lst.append(features_hid)
        pred_m_lst.append(df_m_mh)
        pred_ob_lst.append(df_ob_mh)
    
    pred_features_df = pd.concat(pred_features_lst)

    # Predict and calculate accuracy on the train data
    y_train_pred = best_classifier.predict(X_train)
    train_acc = accuracy_score(y_train_pred, y_train)
    y_train_prob = best_classifier.predict_proba(X_train)[:, 1]
    print("Accuracy on the train data:", train_acc)

    # Set the index of the new DataFrame to match y_test
    y_train_pred_df = pd.DataFrame({'pred_dir': y_train_pred,
                                    'pred_prob': y_train_prob})
    y_train_pred_df.index = y_train.index


    # Predict and calculate accuracy on the test data
    y_test_pred = best_classifier.predict(X_test)
    test_acc = accuracy_score(y_test_pred, y_test)
    y_test_prob = best_classifier.predict_proba(X_test)[:, 1]
    print("Accuracy on the test data:", test_acc)

    # Set the index of the new DataFrame to match y_test
    y_test_pred_df = pd.DataFrame({'pred_dir': y_test_pred,
                                   'pred_prob': y_test_prob})
    y_test_pred_df.index = y_test.index


    # Predict unlabelled data
    y_pred = best_classifier.predict(pred_features_df)
    y_pred_prob = best_classifier.predict_proba(pred_features_df)[:, 1]

    y_pred_df = pd.DataFrame({'pred_dir': y_pred,
                              'pred_prob': y_pred_prob})
    y_pred_df.index = pred_features_df.index

    # Tag direction of hidden liquidity execution outside of bid-ask spread
    y_pred_df = hid_outside_spread_tag(pred_features_df, y_pred_df)

    features_dict = {'labelled': labelled_features,
                     'unlabelled': pred_features_df}
    
    prediction_dict = {'train': y_train_pred_df,
                       'test': y_test_pred_df,
                       'pred': y_pred_df}
    
    df_labelled_dict = {ticker: (df1, df2) for ticker, df1, df2 in zip(tickers_train, labelled_m_lst, labelled_ob_lst)}
    df_predict_dict = {ticker: (df1, df2) for ticker, df1, df2 in zip(tickers_pred, pred_m_lst, pred_ob_lst)}

    return df_labelled_dict, df_predict_dict, features_dict, prediction_dict, best_classifier





# # Define the training pipeline
# training_pipeline = Pipeline([
#     ('preprocessing', DataPreprocessor()),
#     ('feature_creation', FeatureCreator(labelled=True)),
#    # ('classifier', RandomForestClassifier())
# ])

# # Define the prediction pipeline (without the classifier step)
# prediction_pipeline = Pipeline([
#     ('preprocessing', DataPreprocessor()),
#     ('feature_creation', FeatureCreator(labelled=False))
# ])


# df_m_train = pd.read_csv("./data/LOB_2012/MSFT_2012-06-21_34200000_57600000_message_10.csv", header=None)
# df_ob_train = pd.read_csv("./data/LOB_2012/MSFT_2012-06-21_34200000_57600000_orderbook_10.csv", header=None)
# X_train = (df_m_train, df_ob_train)
# training_pipeline.fit(X_train, preprocessing__ticker_name='MSFT')



# #________TRAINING EXAMPLE_____________

# # Example training data (Replace with actual data)
# df_m_train = pd.DataFrame()  # Your training message data here
# df_ob_train = pd.DataFrame()  # Your training order book data here
# X_train = (df_m_train, df_ob_train)  # Input tuple for the training pipeline

# # Fit the preprocessing and feature creation steps
# training_pipeline.named_steps['preprocessing'].fit(X_train, ticker_name='AAPL')
# df_m_mh_train, df_ob_mh_train = training_pipeline.named_steps['preprocessing'].transform(X_train)
# features_train, labels_train = training_pipeline.named_steps['feature_creation'].fit_transform((df_m_mh_train, df_ob_mh_train))

# # Fit the classifier with the extracted features and labels
# training_pipeline.named_steps['classifier'].fit(features_train, labels_train)

# #________PREDICTION EXAMPLE_____________

# df_m_predict = pd.DataFrame()  # Your prediction message data here
# df_ob_predict = pd.DataFrame()  # Your prediction order book data here
# X_predict = (df_m_predict, df_ob_predict)  # Input tuple for the prediction pipeline

# # Fit the preprocessing and feature creation steps for the new ticker
# prediction_pipeline.named_steps['preprocessing'].fit(X_predict, ticker_name='AAPL')
# df_m_mh_predict, df_ob_mh_predict = prediction_pipeline.named_steps['preprocessing'].transform(X_predict)
# features_predict = prediction_pipeline.named_steps['feature_creation'].transform((df_m_mh_predict, df_ob_mh_predict))

# # Predict using the classifier from the training pipeline
# predictions = training_pipeline.named_steps['classifier'].predict(features_predict)

