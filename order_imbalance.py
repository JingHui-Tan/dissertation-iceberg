
'''Compute different versions of conditional Order Imbalance on message and orderbook dataframe'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.formula.api as smf



## Process:
## We fit order_imbalance and then remove outliers based on 90% quantile (for log returns and future log returns)


def iceberg_tag(df, ib_delta):
    '''Add a column tagging whether trade is an iceberg execution'''
    event_type_4 = df[df['event_type'] == 4]
    event_type_1 = df[df['event_type'] == 1]

    # Sort both DataFrames by datetime
    event_type_4 = event_type_4.sort_values(by='datetime')
    event_type_1 = event_type_1.sort_values(by='datetime')

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





def calculate_log_returns(grouped):
    grouped['log_ret'] = np.log(grouped['last_midprice']) - np.log(grouped['first_midprice'])
    grouped['fut_log_ret'] = grouped['log_ret'].shift(-1)
    grouped['weighted_log_ret'] = np.log(grouped['last_weighted_mp']) - np.log(grouped['first_weighted_mp'])
    grouped['fut_weighted_log_ret'] = grouped['weighted_log_ret'].shift(-1)
    return grouped


def filter_quantiles(grouped, column, lower_quantile=0.025, upper_quantile=0.975):
    lower_bound = grouped[column].quantile(lower_quantile)
    upper_bound = grouped[column].quantile(upper_quantile)
    return grouped[(grouped[column] >= lower_bound) & (grouped[column] <= upper_bound)]




def calculate_order_imbalance(df, direction_column, size_column, pred_dir_column=None):
    if pred_dir_column:
        return (df[size_column] * (1 - 2 * df[pred_dir_column])).sum() / df[size_column].sum()
    else:
        buy_size = df.loc[df[direction_column] == -1, size_column].sum()
        sell_size = df.loc[df[direction_column] == 1, size_column].sum()
        return (buy_size - sell_size) / (sell_size + buy_size)





def order_imbalance(df_full, df_pred=None, df_ob=None, delta='30S', type='vis'):
    weight = df_ob['bid_size_1'] / (df_ob['bid_size_1'] + df_ob['ask_size_1'])
    df_full['weighted_mp'] = weight * df_ob['ask_price_1'] + (1 - weight) * df_ob['bid_price_1']

    if type == 'vis':
        df = df_full[df_full['event_type'] == 4]
    elif type == 'hid':
        df = df_full[df_full['event_type'] == 5]
    else:
        print("Not Implemented")
        return
    df['datetime_bins'] = df.index.get_level_values('datetime').ceil(delta)

    if type == 'hid':
        df = pd.merge(df.reset_index(), df_pred.reset_index(), on=['datetime', 'ticker', 'event_number']).set_index(['datetime', 'ticker', 'event_number'])
    grouped = df.groupby('datetime_bins').apply(
        lambda x: pd.Series({
            'order_imbalance': calculate_order_imbalance(x, 'direction', 'size', 'pred_dir' if type == 'hid' else None),
            'first_midprice': x['midprice'].iloc[0],
            'last_midprice': x['midprice'].iloc[-1],
            'first_weighted_mp': x['weighted_mp'].iloc[0],
            'last_weighted_mp': x['weighted_mp'].iloc[-1]
        })
    ).reset_index()

    grouped['order_imbalance'] = grouped['order_imbalance'].fillna(0)
    grouped = calculate_log_returns(grouped)

    grouped = filter_quantiles(grouped, 'log_ret')
    grouped = filter_quantiles(grouped, 'fut_log_ret')
    grouped = filter_quantiles(grouped, 'weighted_log_ret')
    grouped = filter_quantiles(grouped, 'fut_weighted_log_ret')

    
    return grouped[:-1]



def combined_order_imbalance(df_full, df_pred, df_ob, delta='5min'):
    df_vis = order_imbalance(df_full=df_full, df_ob=df_ob, delta=delta, type='vis')
    df_hid = order_imbalance(df_full=df_full, df_ob=df_ob, df_pred=df_pred, delta=delta, type='hid')

    return df_vis.merge(df_hid[['datetime_bins', 'order_imbalance']], on='datetime_bins', suffixes=('_vis', '_hid'))




def iceberg_order_imbalance(df_full, df_pred, df_ob, delta='5min', weighted=False):
    weight = df_ob['bid_size_1'] / (df_ob['bid_size_1'] + df_ob['ask_size_1'])
    df_full['weighted_mp'] = weight * df_ob['ask_price_1'] + (1 - weight) * df_ob['bid_price_1']
    ib_delta = '1ms'
    df = iceberg_tag(df_full, ib_delta)
    df['datetime_bins'] = df.index.get_level_values('datetime').ceil(delta)
    grouped = df.groupby('datetime_bins').apply(
        lambda x: pd.Series({
            'order_imbalance_vis': calculate_order_imbalance(x[x['iceberg'] == 0], 'direction', 'size'),
            'order_imbalance_ib': calculate_order_imbalance(x[x['iceberg'] == 1], 'direction', 'size'),
            'first_midprice': x['midprice'].iloc[0],
            'last_midprice': x['midprice'].iloc[-1],
            'first_weighted_mp': x['weighted_mp'].iloc[0],
            'last_weighted_mp': x['weighted_mp'].iloc[-1]
        })
    ).reset_index()

    grouped['order_imbalance_vis'] = grouped['order_imbalance_vis'].fillna(0)
    grouped['order_imbalance_ib'] = grouped['order_imbalance_ib'].fillna(0)
    grouped = calculate_log_returns(grouped)

    grouped = filter_quantiles(grouped, 'log_ret')
    grouped = filter_quantiles(grouped, 'fut_log_ret')
    grouped = filter_quantiles(grouped, 'weighted_log_ret')
    grouped = filter_quantiles(grouped, 'fut_weighted_log_ret')


    df_hid = order_imbalance(df_full=df_full, df_pred=df_pred, df_ob=df_ob, delta=delta, type='hid')
    df_hid.rename(columns={"order_imbalance": "order_imbalance_hid"}, inplace=True)

    return grouped.merge(df_hid[['datetime_bins', 'order_imbalance_hid']], on='datetime_bins')



def agg_order(df, df_pred, agg='agg_low'):
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
    df = df[df['event_type']==5]
    lower_q = df['size'].quantile(1/3)
    upper_q = df['size'].quantile(2/3)

    if size=='small':
        return df[df['size'] < lower_q]
    
    elif size=='medium':
        return df[(df['size'] > lower_q) & (df['size'] < upper_q)]

    elif size=='large':
        return df[df['size'] > upper_q]



def conditional_order_imbalance(df_full, df_pred, df_ob, delta='5min', condition='agg'):

    df_fin = order_imbalance(df_full=df_full, delta=delta, df_ob=df_ob, type='vis')
    df_fin.rename(columns={'order_imbalance': 'order_imbalance_vis'}, inplace=True)

    if condition == 'agg':
        for version, suffix in zip(['agg_low', 'agg_mid', 'agg_high'], ['_agg_low', '_agg_mid', '_agg_high']):
            df = order_imbalance(df_full=agg_order(df_full, df_pred, agg=version), df_pred=df_pred, df_ob=df_ob, delta=delta, type='hid')
            df.rename(columns={'order_imbalance': f'order_imbalance{suffix}'}, inplace=True)
            df_fin = df_fin.merge(df[['datetime_bins', f'order_imbalance{suffix}']], on='datetime_bins')

    elif condition == 'size':
        for version in ['small', 'medium', 'large']:
            df = order_imbalance(df_full=size_order(df_full, df_pred, size=version), df_pred=df_pred, df_ob=df_ob, delta=delta, type='hid')
            df.rename(columns={'order_imbalance': f'order_imbalance_{version}'}, inplace=True)
            df_fin = df_fin.merge(df[['datetime_bins', f'order_imbalance_{version}']], on='datetime_bins')

    return df_fin



def lm_results(df_full, df_pred, df_ob, delta_lst, order_type='combined', predictive=True, weighted_mp=False,
               momentum=False):

    if weighted_mp==False:
        y = "fut_log_ret" if predictive else "log_ret"
        x_momentum = "+ log_ret" if momentum else ""
    else:
        y = 'fut_weighted_log_ret' if predictive else "weighted_log_ret"
        x_momentum = "+ weighted_log_ret" if momentum else ""
    
    params_lst = []
    tvalues_lst = []

    for delta in delta_lst:
        if order_type == 'vis' or order_type == 'hid' or order_type == 'combined':
            df_merged = combined_order_imbalance(df_full, df_pred, df_ob, delta=delta)

        elif order_type == 'comb_iceberg':
            df_merged = iceberg_order_imbalance(df_full, df_pred, df_ob, delta=delta)
            lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis + order_imbalance_hid + order_imbalance_ib + {x_momentum}""", 
                         data=df_merged).fit()
            params_lst.append((lm.params[1], lm.params[2], lm.params[3]))
            tvalues_lst.append((lm.tvalues[1], lm.tvalues[2], lm.tvalues[3]))

        elif order_type == 'agg':
            df_merged = conditional_order_imbalance(df_full, df_pred, df_ob, delta=delta, condition='agg')
            lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis + order_imbalance_agg_low + \n
                         order_imbalance_agg_mid + order_imbalance_agg_high {x_momentum}""", 
                         data=df_merged).fit()
            params_lst.append((lm.params[1], lm.params[2], lm.params[3], lm.params[4]))
            tvalues_lst.append((lm.tvalues[1], lm.tvalues[2], lm.tvalues[3], lm.tvalues[4]))
        
        elif order_type == 'size':
            df_merged = conditional_order_imbalance(df_full, df_pred, df_ob, delta=delta, condition='size')
            lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis + order_imbalance_small + \n
                         order_imbalance_medium + order_imbalance_large {x_momentum}""", 
                         data=df_merged).fit()
            params_lst.append((lm.params[1], lm.params[2], lm.params[3], lm.params[4]))
            tvalues_lst.append((lm.tvalues[1], lm.tvalues[2], lm.tvalues[3], lm.tvalues[4]))

        if order_type == "vis":
            lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis {x_momentum}""", data=df_merged).fit()
            params_lst.append(lm.params[1])
            tvalues_lst.append(lm.tvalues[1])

        elif order_type == "hid":
            lm = smf.ols(formula=f"""{y} ~ order_imbalance_hid {x_momentum}""", data=df_merged).fit()
            params_lst.append(lm.params[1])
            tvalues_lst.append(lm.tvalues[1])

        elif order_type == "combined":
            lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis + order_imbalance_hid {x_momentum}""", data=df_merged).fit()
            params_lst.append((lm.params[1], lm.params[2]))
            tvalues_lst.append((lm.tvalues[1], lm.tvalues[2]))


    if order_type == "vis" or order_type == "hid":
        return pd.DataFrame({"timeframe": delta_lst, "params": params_lst, "tvalues": tvalues_lst})

    elif order_type == "combined":
        df = pd.DataFrame({
            'timeframe': delta_lst,
            'params_vis': [x[0] for x in params_lst],
            'tvalues_vis': [x[0] for x in tvalues_lst],
            'params_hid': [x[1] for x in params_lst],
            'tvalues_hid': [x[1] for x in tvalues_lst]
        })
        return df


    elif order_type == 'comb_iceberg':
        df = pd.DataFrame({
            'timeframe': delta_lst,
            'params_vis': [x[0] for x in params_lst],
            'tvalues_vis': [x[0] for x in tvalues_lst],
            'params_hid': [x[1] for x in params_lst],
            'tvalues_hid': [x[1] for x in tvalues_lst],
            'params_ib': [x[2] for x in params_lst],
            'tvalues_ib': [x[2] for x in tvalues_lst]
        })
        return df


    elif order_type == 'agg':
        df = pd.DataFrame({
            'timeframe': delta_lst,
            'params_vis': [x[0] for x in params_lst],
            'tvalues_vis': [x[0] for x in tvalues_lst],
            'params_agg_low': [x[1] for x in params_lst],
            'tvalues_agg_low': [x[1] for x in tvalues_lst],
            'params_agg_mid': [x[2] for x in params_lst],
            'tvalues_agg_mid': [x[2] for x in tvalues_lst],
            'params_agg_high': [x[3] for x in params_lst],
            'tvalues_agg_high': [x[3] for x in tvalues_lst],
        })
        return df
    
    elif order_type == 'size':
        df = pd.DataFrame({
            'timeframe': delta_lst,
            'params_vis': [x[0] for x in params_lst],
            'tvalues_vis': [x[0] for x in tvalues_lst],
            'params_small': [x[1] for x in params_lst],
            'tvalues_small': [x[1] for x in tvalues_lst],
            'params_medium': [x[2] for x in params_lst],
            'tvalues_medium': [x[2] for x in tvalues_lst],
            'params_large': [x[3] for x in params_lst],
            'tvalues_large': [x[3] for x in tvalues_lst],
        })
        return df


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
