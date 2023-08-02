# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:40:01 2023

@author: KenYew Chan
"""

# Double-Strategy (Min Optimisation) with Rolling MPT

import pandas as pd
import numpy as np
import yfinance as yf
import pyfolio as pf
from talib import ATR
from talib import TSF
import optuna
import matplotlib.pyplot as plt
from pyfolio import timeseries as pf

from numpy import *
from numpy.linalg import multi_dot

# Import cufflinks
import cufflinks as cf
cf.set_config_file(offline=True, dimensions=((1000,600))) # theme= 'henanigans'

# Import plotly express for EF plot
import plotly.express as px
# px.defaults.template = "plotly_dark"
px.defaults.width, px.defaults.height = 1000, 600

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.precision', 3)

# Define Time Frames

# BackTest Period
start1 = "2018-07-01"
end1 = "2022-12-31"

# Forward Test Period
start2 = "2022-07-01"
end2 = "2023-12-31"

def Z_Score(df, window):
    df['mean'] = df['PremDisc%'].rolling(window=window).mean()
    df['std'] = df['PremDisc%'].rolling(window=window).std()
    df['Z_Score'] = (df['PremDisc%'] - df['mean']) / df['std']
    return df


def ZScore(df, window):
    df['Mean'] = df['MisPricing%'].rolling(window=window).mean()
    df['Std'] = df['MisPricing%'].rolling(window=window).std()
    df['ZScore'] = (df['MisPricing%'] - df['Mean']) / df['Std']
    return df


def calculate_strategy_return(df, params):

    df["SMA"] = df["Close"].rolling(window=params['sma_window']).mean()
    df["PremDisc%"] = (df["Close"]/df["SMA"] - 1)*100
    df = Z_Score(df, window=params['zscore_window'])

    df["TSF"] = TSF(df["Close"].values, timeperiod=params['sma_window'])
    df["MisPricing%"] = (df["TSF"]/df["Close"] - 1)*100
    df = ZScore(df, window=params['zscore_window'])
    
    buyZone1 = (df["Z_Score"] > -1*params['buyZScoreTH']) & (df["ZScore"] > -1*params['buyZScoreTH'])
    prebuyZone1 = (df["Z_Score"].shift(1) <= -1*params['buyZScoreTH']) & (df["ZScore"].shift(1) <= -1*params['buyZScoreTH'])
    buyZone2 = (df["Z_Score"] > 1*0.25) & (df["ZScore"] > 1*0.25)
    prebuyZone2 = (df["Z_Score"].shift(1) <= 1*0.25) & (df["ZScore"].shift(1) <= 1*0.25)
    sellZone1 = (df["Z_Score"] < 1*params['sellZScoreTH']) & (df["ZScore"] < 1*params['sellZScoreTH'])
    presellZone1 = (df["Z_Score"].shift(1) >= 1*params['sellZScoreTH']) & (df["ZScore"].shift(1) >= 1*params['sellZScoreTH'])
    sellZone2 = ((df["Z_Score"] < 0) & (df["Z_Score"].shift(1) > 0)) | ((df["ZScore"] < 0) & (df["ZScore"].shift(1) > 0))

    df["BuySignal"] = np.where((buyZone1 & prebuyZone1) | (buyZone2 & prebuyZone2), 1, 0)
    df["SellSignal"] = np.where((sellZone1 & presellZone1) | sellZone2, -1, 0)
    df["Signal"] = df["BuySignal"] + df["SellSignal"]
    
    df["ATR"] = ATR(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=params['ATR_window'])
    
    df["SL"] = np.where(df["Signal"]==1, df["Close"] - params['SL_multiplier'] * df["ATR"], np.nan)
    df["TP"] = np.where(df["Signal"]==1, df["Close"] + params['TP_multiplier'] * df["ATR"], np.nan)
    df["SL"].fillna(method='ffill', inplace=True)
    df["TP"].fillna(method='ffill', inplace=True)
    
    df['returns'] = df['Close'].pct_change() * df['Signal'].shift()
    
    return df['returns']


def objective(trial, df):
    
    params = {
        'sma_window': trial.suggest_int('sma_window', 233, 233),
        'zscore_window': trial.suggest_int('zscore_window', 13, 13),
        'buyZScoreTH': trial.suggest_float('buyZScoreTH', 2.1, 2.1),
        'sellZScoreTH': trial.suggest_float('sellZScoreTH', 2.1, 2.1),
        'SL_multiplier': trial.suggest_float('SL_multiplier', 1.5, 1.5),
        'TP_multiplier': trial.suggest_float('TP_multiplier', 3.0, 3.0),
        'ATR_window': trial.suggest_int('ATR_window', 13, 13)
    }
    returns = calculate_strategy_return(df.copy(), params)
    sharpe_ratio = pf.sharpe_ratio(returns.dropna())
    
    return sharpe_ratio

tickers = ["1961.KL", "1015.KL", "1023.KL", "1066.KL", "5246.KL", "5681.KL", "5347.KL", "1155.KL", "3816.KL",
           "5183.KL", "4863.KL", "3182.KL", "1295.KL", "6012.KL", "2445.KL", "5225.KL", "4065.KL", "4707.KL", 
           "4715.KL", "8869.KL", "1082.KL", "5285.KL", "6033.KL", "7084.KL", "5819.KL", "6947.KL", "4197.KL", 
           "5296.KL", "7277.KL", "6888.KL", "7113.KL", "0138.KL", "5168.KL", "5184.KL", "6971.KL", "0128.KL"]

unique_tickers = list(set(tickers))

tickers = unique_tickers

returns_df = pd.DataFrame()
best_params_df = pd.DataFrame()
highest_sharpe_ratios = {}


for ticker in tickers:
    try:
        
        df = yf.download(ticker, start=start1, end=end1)
        df["Volume"] = 1 / (df["Adj Close"] / df["Close"]) * df["Volume"] 
        df["Open"] = df["Adj Close"] / df["Close"] * df["Open"] 
        df["High"] = df["Adj Close"] / df["Close"] * df["High"] 
        df["Low"] = df["Adj Close"] / df["Close"] * df["Low"] 
        df["Close"] = df["Adj Close"]

        df = df[df['Volume'] != 0]
        df.fillna(method='ffill', inplace=True)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, df), n_trials=1)
        best_params = study.best_params
        print(f"Best params for {ticker}: {best_params}")
        returns = calculate_strategy_return(df, best_params)
        returns_df[ticker] = returns
        best_params_df = best_params_df.append(pd.Series(best_params, name=ticker))
        highest_sharpe_ratios[ticker] = study.best_value

    except Exception as e:
        print(ticker)
        
equity_curve = (1 + returns_df).cumprod()

highest_sharpe_ratios_df = pd.DataFrame.from_dict(highest_sharpe_ratios, orient='index', columns=['Highest Sharpe Ratio'])

# Combine best_params_df and highest_sharpe_ratios_df into a single dataframe
combined_df = pd.concat([best_params_df, highest_sharpe_ratios_df], axis=1)

# Write combined_df to a CSV file
combined_df.to_csv("min_opt_best_params.csv")

output_df = pd.concat([returns_df, highest_sharpe_ratios_df, equity_curve], axis=1)

print(output_df)

top_tickers_df = highest_sharpe_ratios_df.sort_values('Highest Sharpe Ratio', ascending=False).head(15)
#bottom_tickers_df = highest_sharpe_ratios_df.sort_values('Highest Sharpe Ratio', ascending=False).tail(15)
print(top_tickers_df)
#print(bottom_tickers_df)

#tickers_df = pd.concat([top_tickers_df, bottom_tickers_df], axis=1)
## First, ensure that your list has unique tickers only
## unique_tickers = list(set(tickers))

# Calculate the number of tickers to trim
#num_tickers = len(tickers_df)
#num_to_trim = int(num_tickers * 0.35)

## Sort the tickers if needed, though in case of ticker symbols it might not make much sense
## unique_tickers.sort()

# Create a new list that excludes the top and bottom 45%
#tickers_df = tickers_df[num_to_trim:-num_to_trim]
#print(tickers_df)

# Run the Back-Test

tickers = top_tickers_df.index # tickers_df.index #


# Load the best parameters
best_params = pd.read_csv('min_opt_best_params.csv', index_col=0)  # Assuming ticker symbols are in the first column


for ticker in tickers:
    try:
        # Define a function to calculate the Z-Score
        def Z_Score(df, window):
            df['mean'] = df['PremDisc%'].rolling(window=window).mean()
            df['std'] = df['PremDisc%'].rolling(window=window).std()
            df['Z_Score'] = (df['PremDisc%'] - df['mean']) / df['std']
            return df
        
        def ZScore(df, window):
            df['Mean'] = df['MisPricing%'].rolling(window=window).mean()
            df['Std'] = df['MisPricing%'].rolling(window=window).std()
            df['ZScore'] = (df['MisPricing%'] - df['Mean']) / df['Std']
            return df

        # Fetch the data
        df = yf.download(ticker, start=start1, end=end2)

        df["Volume"] = 1 / (df["Adj Close"] / df["Close"]) * df["Volume"] 
        df["Open"] = df["Adj Close"] / df["Close"] * df["Open"] 
        df["High"] = df["Adj Close"] / df["Close"] * df["High"] 
        df["Low"] = df["Adj Close"] / df["Close"] * df["Low"] 
        df["Close"] = df["Adj Close"]
        df = df[df['Volume'] != 0]
        df.fillna(method='ffill', inplace=True)
        print(df)

        # Get the best parameters for this ticker
        params = best_params.loc[ticker]

        # Calculate the SMA
        df["SMA"] = df["Close"].rolling(window=int(params['sma_window'])).mean()

        # Calculate the premium/discount
        df["PremDisc%"] = (df["Close"]/df["SMA"] - 1)*100

        # Calculate Z-Score
        df = Z_Score(df, window=int(params['zscore_window']))

        df["TSF"] = TSF(df["Close"].values, timeperiod=int(params['sma_window']))
        df["MisPricing%"] = (df["TSF"]/df["Close"] - 1)*100
        df = ZScore(df, window=int(params['zscore_window']))

        
        # Define buy and sell zones
        buyZScoreTH = params['buyZScoreTH']
        sellZScoreTH = params['sellZScoreTH']

        buyZone1 = (df["Z_Score"] > -1*buyZScoreTH) & (df["ZScore"] > -1*buyZScoreTH)
        prebuyZone1 = (df["Z_Score"].shift(1) <= -1*buyZScoreTH) & (df["ZScore"].shift(1) <= -1*buyZScoreTH)
        buyZone2 = (df["Z_Score"] > 1*0.25) & (df["ZScore"] > 1*0.25)
        prebuyZone2 = (df["Z_Score"].shift(1) <= 1*0.25) & (df["ZScore"].shift(1) <= 1*0.25)
        sellZone1 = (df["Z_Score"] < 1*sellZScoreTH) & (df["ZScore"] < 1*sellZScoreTH)
        presellZone1 = (df["Z_Score"].shift(1) >= 1*sellZScoreTH) & (df["ZScore"].shift(1) >= 1*sellZScoreTH)
        sellZone2 = ((df["Z_Score"] < 0) & (df["Z_Score"].shift(1) > 0)) | ((df["ZScore"] < 0) & (df["ZScore"].shift(1) > 0))


        # Generate trading signals
        df["BuySignal"] = np.where((buyZone1 & prebuyZone1) | (buyZone2 & prebuyZone2), 1, 0)
        df["SellSignal"] = np.where((sellZone1 & presellZone1) | sellZone2, -1, 0)
        df["Signal"] = df["BuySignal"] + df["SellSignal"]

        # Calculate the ATR
        df["ATR"] = ATR(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=int(params['ATR_window']))

        # Define SL and TP multipliers
        SL_multiplier = params['SL_multiplier']
        TP_multiplier = params['TP_multiplier']

        # Initialize SL and TP levels to NaN
        df["SL"] = np.nan
        df["TP"] = np.nan

        # Define initial capital
        capital = 10000

        # Calculate the number of shares to buy
        df["Shares"] = capital // (df["ATR"] * 100)

        # Create a variable to hold the entry price
        entry_price = 0

        # Define SL and TP levels based on entry price
        for i in range(len(df)):
            if df["Signal"].iloc[i] == 1:
                entry_price = df["Close"].iloc[i]
                df["SL"].iloc[i] = entry_price - SL_multiplier * df["ATR"].iloc[i]
                df["TP"].iloc[i] = entry_price + TP_multiplier * df["ATR"].iloc[i]
            elif df["Signal"].iloc[i] == -1:
                entry_price = 0
                df["SL"].iloc[i] = np.nan
                df["TP"].iloc[i] = np.nan
            else:
                df["SL"].iloc[i] = df["SL"].iloc[i-1]
                df["TP"].iloc[i] = df["TP"].iloc[i-1]

        # Fill in the rest of SL and TP columns
        df["SL"].fillna(method='ffill', inplace=True)
        df["TP"].fillna(method='ffill', inplace=True)

        # Calculate daily returns based on trading signals
        df['returns'] = df['Close'].pct_change() * df['Signal'].shift()

        # Calculate Buy-and-Hold Returns
        df["B&H_Returns"] = df["Close"].pct_change()

        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod()

        # Calculate buy-and-hold cumulative returns
        df["B&H_Cumulative_Returns"] = (1 + df["B&H_Returns"]).cumprod()

        # Print the DataFrame
        print(df)

        # Plot cumulative returns
        plt.figure(figsize=(13,8))

        # Plot the strategy's cumulative returns
        plt.plot(df['cumulative_returns'], label='Strategy Returns')

        # Plot the buy-and-hold cumulative returns
        plt.plot(df["B&H_Cumulative_Returns"], label='Buy & Hold Returns')

        plt.title('Cumulative Returns of the Strategy vs Buy & Hold')
        plt.legend(loc='best')
        plt.show()
        
    except Exception as e:
        print(ticker)

# Create an empty dataframe to store equity curves
equity_curve_df = pd.DataFrame()

# Create an empty dataframe to store buy and hold equity curves
bh_equity_curve_df = pd.DataFrame()

# Loop through the tickers
for ticker in tickers:
    try:
        
        # Define a function to calculate the Z-Score
        def Z_Score(df, window):
            df['mean'] = df['PremDisc%'].rolling(window=window).mean()
            df['std'] = df['PremDisc%'].rolling(window=window).std()
            df['Z_Score'] = (df['PremDisc%'] - df['mean']) / df['std']
            return df
        
        def ZScore(df, window):
            df['Mean'] = df['MisPricing%'].rolling(window=window).mean()
            df['Std'] = df['MisPricing%'].rolling(window=window).std()
            df['ZScore'] = (df['MisPricing%'] - df['Mean']) / df['Std']
            return df


        # Fetch the data
        df = yf.download(ticker, start=start1, end=end2)

        df["Volume"] = 1 / (df["Adj Close"] / df["Close"]) * df["Volume"] 
        df["Open"] = df["Adj Close"] / df["Close"] * df["Open"] 
        df["High"] = df["Adj Close"] / df["Close"] * df["High"] 
        df["Low"] = df["Adj Close"] / df["Close"] * df["Low"] 
        df["Close"] = df["Adj Close"]
        df = df[df['Volume'] != 0]
        df.fillna(method='ffill', inplace=True)
        print(df)

        # Get the best parameters for this ticker
        params = best_params.loc[ticker]

        # Calculate the SMA
        df["SMA"] = df["Close"].rolling(window=int(params['sma_window'])).mean()

        # Calculate the premium/discount
        df["PremDisc%"] = (df["Close"]/df["SMA"] - 1)*100

        # Calculate Z-Score
        df = Z_Score(df, window=int(params['zscore_window']))
        
        df["TSF"] = TSF(df["Close"].values, timeperiod=int(params['sma_window']))
        df["MisPricing%"] = (df["TSF"]/df["Close"] - 1)*100
        df = ZScore(df, window=int(params['zscore_window']))


        # Define buy and sell zones
        buyZScoreTH = params['buyZScoreTH']
        sellZScoreTH = params['sellZScoreTH']

        buyZone1 = (df["Z_Score"] > -1*buyZScoreTH) & (df["ZScore"] > -1*buyZScoreTH)
        prebuyZone1 = (df["Z_Score"].shift(1) <= -1*buyZScoreTH) & (df["ZScore"].shift(1) <= -1*buyZScoreTH)
        buyZone2 = (df["Z_Score"] > 1*0.25) & (df["ZScore"] > 1*0.25)
        prebuyZone2 = (df["Z_Score"].shift(1) <= 1*0.25) & (df["ZScore"].shift(1) <= 1*0.25)
        sellZone1 = (df["Z_Score"] < 1*sellZScoreTH) & (df["ZScore"] < 1*sellZScoreTH)
        presellZone1 = (df["Z_Score"].shift(1) >= 1*sellZScoreTH) & (df["ZScore"].shift(1) >= 1*sellZScoreTH)
        sellZone2 = ((df["Z_Score"] < 0) & (df["Z_Score"].shift(1) > 0)) | ((df["ZScore"] < 0) & (df["ZScore"].shift(1) > 0))

        # Generate trading signals
        df["BuySignal"] = np.where((buyZone1 & prebuyZone1) | (buyZone2 & prebuyZone2), 1, 0)
        df["SellSignal"] = np.where((sellZone1 & presellZone1) | sellZone2, -1, 0)
        df["Signal"] = df["BuySignal"] + df["SellSignal"]

        # Calculate the ATR
        df["ATR"] = ATR(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=int(params['ATR_window']))

        # Define SL and TP multipliers
        SL_multiplier = params['SL_multiplier']
        TP_multiplier = params['TP_multiplier']

        # Initialize SL and TP levels to NaN
        df["SL"] = np.nan
        df["TP"] = np.nan

        # Define initial capital
        capital = 10000

        # Calculate the number of shares to buy
        df["Shares"] = capital // (df["ATR"] * 100)

        # Create a variable to hold the entry price
        entry_price = 0

        # Define SL and TP levels based on entry price
        for i in range(len(df)):
            if df["Signal"].iloc[i] == 1:
                entry_price = df["Close"].iloc[i]
                df["SL"].iloc[i] = entry_price - SL_multiplier * df["ATR"].iloc[i]
                df["TP"].iloc[i] = entry_price + TP_multiplier * df["ATR"].iloc[i]
            elif df["Signal"].iloc[i] == -1:
                entry_price = 0
                df["SL"].iloc[i] = np.nan
                df["TP"].iloc[i] = np.nan
            else:
                df["SL"].iloc[i] = df["SL"].iloc[i-1]
                df["TP"].iloc[i] = df["TP"].iloc[i-1]

        # Fill in the rest of SL and TP columns
        df["SL"].fillna(method='ffill', inplace=True)
        df["TP"].fillna(method='ffill', inplace=True)

        # Calculate daily returns based on trading signals
        df['returns'] = df['Close'].pct_change() * df['Signal'].shift()

        # Calculate Buy-and-Hold Returns
        df["B&H_Returns"] = df["Close"].pct_change()

        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod()

        # Calculate buy-and-hold cumulative returns
        df["B&H_Cumulative_Returns"] = (1 + df["B&H_Returns"]).cumprod()

        # Print the DataFrame
        print(df)

        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        
        # Calculate buy-and-hold cumulative returns
        df["B&H_Cumulative_Returns"] = (1 + df["B&H_Returns"]).cumprod()

        # Add the cumulative returns to the equity curve dataframe
        equity_curve_df[ticker+'_Strategy'] = df['cumulative_returns']

        # Add the buy-and-hold cumulative returns to the buy and hold equity curve dataframe
        bh_equity_curve_df[ticker+'_B&H'] = df["B&H_Cumulative_Returns"]

        # Plot cumulative returns
        plt.figure(figsize=(13,8))

        # Plot the strategy's cumulative returns
        plt.plot(df['cumulative_returns'], label='Strategy Returns')

        # Plot the buy-and-hold cumulative returns
        plt.plot(df["B&H_Cumulative_Returns"], label='Buy & Hold Returns')

        plt.title('Cumulative Returns of the Strategy vs Buy & Hold')
        plt.legend(loc='best')
        plt.show()
     
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Print the equity curve dataframe
print(equity_curve_df)

# Print the buy and hold equity curve dataframe
print(bh_equity_curve_df)

# Concat and Print Both the DataFrames
final_df = pd.concat([equity_curve_df, bh_equity_curve_df], axis=1)
print(final_df)

# MPT using  Monte Carlo simulation to generate a large number of random portfolios
# (5,000 in this case) and then selects the portfolio with the highest Sharpe Ratio. 
# This is a simpler and more brute-force approach, and it may not always find the global 
# optimum, especially if the number of portfolios simulated is not large enough.
# This method can be more intuitive and easier to understand, and it can sometimes find better 
# solutions, especially for complex or non-convex optimization problems. However, it can also 
# be slower and less accurate, especially for larger numbers of assets or smaller numbers of 
# simulations.

# Specify assets / stocks
assets = final_df.columns

# Number of assets
num_of_assets = len(assets)

# Number of portfolio for optimization
num_of_portfolios = 5000

# Source of Data or Equity Curve of ALL Tickers
df = final_df
df = df.fillna(method='ffill')
#df = df.fillna(method='bfill')
df.dropna(inplace=True)

# Define the rolling window size
rolling_window = int(252)  # You can adjust this based on your preference (e.g., 1 year)

# Initialize a DataFrame to store the rolling portfolio weights
rolling_weights_df = pd.DataFrame(index=final_df.index, columns=assets)

# Perform rolling optimization of the portfolio
for i in range(rolling_window, len(final_df)):
    window_returns = final_df.iloc[i - rolling_window:i]  # Get the returns for the rolling window

    # Define the function for portfolio simulation with random weights
    def portfolio_simulation(returns):
        weights = np.random.random(num_of_assets)
        weights /= np.sum(weights)
        port_returns = np.sum(weights * returns.mean() * 252)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = port_returns / port_volatility
        return weights, sharpe_ratio

    # Perform Monte Carlo simulation to find the maximum Sharpe ratio portfolio for the rolling window
    max_sharpe_ratio = -np.inf
    best_weights = None
    for j in range(num_of_portfolios):
        weights, sharpe_ratio = portfolio_simulation(window_returns)
        if sharpe_ratio > max_sharpe_ratio:
            max_sharpe_ratio = sharpe_ratio
            best_weights = weights

    # Store the weights of the maximum Sharpe ratio portfolio for the rolling window
    rolling_weights_df.iloc[i] = best_weights

# Forward fill any NaN values in the rolling weights DataFrame
rolling_weights_df.fillna(method='ffill', inplace=True)

# Calculate the portfolio returns based on the rolling weights
rolling_portfolio_returns = (rolling_weights_df.shift() * final_df.pct_change()).sum(axis=1)

# Calculate the cumulative returns of the rolling portfolio
cumulative_returns = (1 + rolling_portfolio_returns).cumprod()

# Print the DataFrame with the rolling portfolio returns and cumulative returns
rolling_portfolio_df = pd.DataFrame({
    'Rolling_Portfolio_Returns': rolling_portfolio_returns,
    'Cumulative_Returns': cumulative_returns
})
print(rolling_portfolio_df)


# Plot the cumulative returns of the portfolio
plt.figure(figsize=(13, 8))
plt.plot(rolling_portfolio_df['Cumulative_Returns'])
plt.title('Cumulative Returns of the Portfolio Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(False)
plt.show()

# Plot the portfolio weights over time
rolling_weights_df.plot(kind='area', stacked=True, figsize=(13, 8))
plt.title('Portfolio Weights Over Time')
plt.xlabel('Date')
plt.ylabel('Weights')
plt.grid(False)
plt.show()

# Print the final portfolio weights
print('Final portfolio weights:')
print(rolling_weights_df.iloc[-1])