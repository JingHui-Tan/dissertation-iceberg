# Hidden Liquidity: Price Impact and Trading Strategies

## Introduction

This repository contains the code for my dissertation: **Hidden Liquidity: Price Impact and Trading Strategies**. The project explores the price impact of hidden liquidity in NASDAQ, using Limit Order Book data from [LOBSTER](https://lobsterdata.com/). In this repository, we present the key files needed for the main components of the disertation:

- Exploratory Data Analysis
- Prediction of Hidden order execution (HoE) directions
- Contemporaneous and predictive regressions for impact of conditional order imbalance
- Trading strategies using hidden liquidity signals

## Data
The analysis relies on the [LOBSTER dataset](https://lobsterdata.com/), which contains detailed limit order book data. This includes the messages and orderbook dataframes which highlight each event, on a ticker-level, occuring throughout the trading day.

## Key Files

The file `chunk_pipeline.py` consolidates all processes, including data preprocessing, implementing machine learning predictions, and calculating linear regression coefficients, using functions from other files. The breakdown is as follows:

- The primary file for preprocessing LOBSTER data and extracting features required for predicting the direction of Hidden Order Executions (HoEs) is `prediction_ML_pipeline.py`.

- The configuration, training and predicting of the XGBoost model is in `chunk_pipeline.py`. A separate implementation exists in `prediction_ML_pipeline.py`.

- Intraday and daily linear regression calculations are done in `chunk_pipeline.py` and `ClOp_calc.py` respectively.

- Order imbalance and return calculations are done in `order_imbalance.py`.

Apart from that, the file `trading_strategy.py` contains all the functions used for the single stock and portfolio strategies.

Additioanlly, the files ending with `_workflow.py` are files used to run different processes on the server. They streamline the aggregation of these processes, allowing for easy modification of configurations to run different variations efficiently.

Finally, the remaining files are either used for exploratory data analysis (EDA), such as generating plots and summary statistics, or for analyzing the dataframes obtained from regressions or strategy outputs.