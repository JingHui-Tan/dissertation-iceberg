
'''Compute different versions of conditional Order Imbalance on message and orderbook dataframe'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.formula.api as smf



## Process:
## We fit order_imbalance and then remove outliers (for log returns)


def iceberg_tag(df, ib_delta):
    '''Add a column tagging whether trade is an iceberg execution'''
    event_type_4 = df[df['event_type'] == 4]
    event_type_1 = df[df['event_type'] == 1]

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




def order_imbalance(df_full, df_pred=None, delta='30S', type='vis'):

    # Obtain visible order imbalance
    if type == 'vis':
        df = df_full[df_full['event_type'] == 4]
        df['datetime_bins'] = df.index.get_level_values('datetime').ceil(delta)

        # Group by the delta intervals and calculate the sums for each direction
        grouped = df.groupby('datetime_bins').apply(
            lambda x: pd.Series({
                'order_imbalance_vis': (x.loc[x['direction'] == -1, 'size'].sum() - x.loc[x['direction'] == 1, 'size'].sum()) /
                                (x.loc[x['direction'] == -1, 'size'].sum() + x.loc[x['direction'] == 1, 'size'].sum()),
                'first_midprice': x['midprice'].iloc[0],
                'last_midprice': x['midprice'].iloc[-1]
            })
        ).reset_index()
        grouped['order_imbalance_vis'] = grouped['order_imbalance_vis'].fillna(0)   

    # Obtain hidden order imbalance
    elif type == 'hid':
        df_hid_trade = df_full[df_full['event_type'] == 5]
        df_hid_trade['datetime_bins'] = df_hid_trade.index.get_level_values('datetime').ceil(delta)

        # Resetting index to allow for merge on multiindex
        df_hid_trade_reset = df_hid_trade.reset_index()
        df_pred_reset = df_pred.reset_index()

        # Merging the DataFrames on the multiindex
        df_merged = pd.merge(df_hid_trade_reset, df_pred_reset, on=['datetime', 'ticker', 'event_number'])

        # Set the multiindex back if necessary
        df_merged.set_index(['datetime', 'ticker', 'event_number'], inplace=True)

        grouped = df_merged.groupby('datetime_bins').apply(
            lambda x: pd.Series({
                'order_imbalance_hid': (x['size'] * (1 - 2 * x['pred_dir'])).sum() / x['size'].sum(),
                'first_midprice': x['midprice'].iloc[0],
                'last_midprice': x['midprice'].iloc[-1]
            })
        ).reset_index()
        
        grouped['order_imbalance_hid'] = grouped['order_imbalance_hid'].fillna(0)

    else:
        print("Not Implemented")
        pass
 
    grouped['log_ret'] = np.log(grouped['last_midprice']) - np.log(grouped['first_midprice'])
    grouped['fut_log_ret'] = grouped['log_ret'].shift(-1)

    # Calculate the 2.5 and 97.5 quantiles
    lower_quantile = grouped['log_ret'].quantile(0.025)
    upper_quantile = grouped['log_ret'].quantile(0.975)

    lower_quantile_fut = grouped['fut_log_ret'].quantile(0.025)
    upper_quantile_fut = grouped['fut_log_ret'].quantile(0.975)


    # Filter the DataFrame
    grouped_filtered = grouped[(grouped['log_ret'] >= lower_quantile) & (grouped['log_ret'] <= upper_quantile)]
    grouped_filtered = grouped_filtered[(grouped_filtered['fut_log_ret'] >= lower_quantile_fut) 
                               & (grouped_filtered['fut_log_ret'] <= upper_quantile_fut)]
    
    grouped_filtered = grouped_filtered[:-1]

    return grouped_filtered



def combined_order_imbalance(df_full, df_pred, delta='5min'):
    # Create combined order_imbalance
    df_vis = order_imbalance(df_full=df_full, delta=delta, type='vis')
    df_hid = order_imbalance(df_full=df_full, df_pred=df_pred, delta=delta, type='hid')

    df_merged = df_vis.merge(df_hid['order_imbalance_hid'], left_index=True, right_index=True)
    return df_merged
    


def conditional_order_imbalance(hidden=True):
    pass



def lm_results(df_full, df_pred, delta_lst, order_type='combined', predictive=True):

    y = "fut_log_ret" if predictive else "log_ret"
    
    params_lst = []
    tvalues_lst = []

    for delta in delta_lst:
        df_merged = combined_order_imbalance(df_full, df_pred, delta=delta)

        if order_type == "vis":
            lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis""", data=df_merged).fit()
            params_lst.append(lm.params[1])
            tvalues_lst.append(lm.tvalues[1])

        elif order_type == "hid":
            lm = smf.ols(formula=f"""{y} ~ order_imbalance_hid""", data=df_merged).fit()
            params_lst.append(lm.params[1])
            tvalues_lst.append(lm.tvalues[1])

        elif order_type == "combined":
            lm = smf.ols(formula=f"""{y} ~ order_imbalance_vis + order_imbalance_hid""", data=df_merged).fit()
            params_lst.append((lm.params[1], lm.params[2]))
            tvalues_lst.append((lm.tvalues[1], lm.tvalues[2]))

    if order_type == "vis" or order_type == "vis":
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
