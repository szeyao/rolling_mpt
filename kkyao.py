import pandas as pd
import MetaTrader5 as mt5
import numpy as np
import talib as tb

def func_MT5_POSITIONS_GET():
    # open positions
    positions=mt5.positions_get()
    # If currently no positions
    if (type(positions)==None):
        # return an empty dataframe
        return pd.DataFrame([], columns=['ticket','time','type','magic','identifier','reason','volume','price_open','sl','tp','price_current','swap','profit','symbol','comment'])
    else:
        if len(positions)==0:
            # return an empty dataframe
            return pd.DataFrame([], columns=['ticket','time','type','magic','identifier','reason','volume','price_open','sl','tp','price_current','swap','profit','symbol','comment'])
        elif len(positions)>0:
            df_positions=pd.DataFrame(list(positions),columns=positions[0]._asdict().keys())
            df_positions['time'] = pd.to_datetime(df_positions['time'], unit='s')
            df_positions.drop(['time_update', 'time_msc', 'time_update_msc', 'external_id'], axis=1, inplace=True)
            return df_positions        
        else:
            raise ValueError('Unaccounted condiiton when getting open positions')


def download_mt5_data(mtstock, time_frame, endBar, n_datapoint, prefix=None):
    stock_data = pd.DataFrame(mt5.copy_rates_from_pos(mtstock, time_frame, endBar, n_datapoint))
    stock_data = stock_data[['time', 'open', 'high', 'low', 'close', 'real_volume']]
    stock_data["time"] = pd.to_datetime(stock_data["time"],unit="s")
    # Apply prefix to column names if provided
    if prefix:
        stock_data.rename(columns = {'time':f'{prefix}Date', 'open':f'{prefix}Open', 
                                     'high':f'{prefix}High', 'low':f'{prefix}Low', 
                                     'close':f'{prefix}Close', 'real_volume':f'{prefix}Volume'}, inplace = True)
        stock_data.set_index(f'{prefix}Date', inplace = True)
    else:
        stock_data.rename(columns = {'time':'Date', 'open':'Open', 
                                     'high':'High', 'low':'Low', 
                                     'close':'Close', 'real_volume':'Volume'}, inplace = True)
        stock_data.set_index('Date', inplace = True)
    return stock_data


def supertrend(df, period=13, multiplier=2.5):
    hl2 = (df['High'] + df['Low']) / 2
    atr = tb.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
    
    # Compute the basic upper and lower bands
    df['upper_band_basic'] = hl2 + (multiplier * atr)
    df['lower_band_basic'] = hl2 - (multiplier * atr)
    
    # Compute the final upper and lower bands
    df['upper_band'] = 0.0
    df['lower_band'] = 0.0
    df['upper_band'][period-1] = df['upper_band_basic'][period-1]
    df['lower_band'][period-1] = df['lower_band_basic'][period-1]
    
    for i in range(period, len(df)):
        if df['Close'][i-1] <= df['upper_band'][i-1]:
            df['upper_band'][i] = min(df['upper_band_basic'][i], df['upper_band'][i-1])
        else:
            df['upper_band'][i] = df['upper_band_basic'][i]
        
        if df['Close'][i-1] >= df['lower_band'][i-1]:
            df['lower_band'][i] = max(df['lower_band_basic'][i], df['lower_band'][i-1])
        else:
            df['lower_band'][i] = df['lower_band_basic'][i]
    
    # Compute the SuperTrend
    df['supertrend'] = 0.0
    df['supertrend'][period-1] = df['upper_band'][period-1]
    for i in range(period, len(df)):
        if df['supertrend'][i-1] == df['upper_band'][i-1] and df['Close'][i] <= df['upper_band'][i]:
            df['supertrend'][i] = df['upper_band'][i]
        elif df['supertrend'][i-1] == df['upper_band'][i-1] and df['Close'][i] > df['upper_band'][i]:
            df['supertrend'][i] = df['lower_band'][i]
        elif df['supertrend'][i-1] == df['lower_band'][i-1] and df['Close'][i] >= df['lower_band'][i]:
            df['supertrend'][i] = df['lower_band'][i]
        elif df['supertrend'][i-1] == df['lower_band'][i-1] and df['Close'][i] < df['lower_band'][i]:
            df['supertrend'][i] = df['upper_band'][i]
    
    # Compute the SuperTrend signals
    df['ST_signal'] = 0
    df.loc[df['Close'] > df['supertrend'], 'ST_signal'] = 1
    df.loc[df['Close'] < df['supertrend'], 'ST_signal'] = -1
    
    # Clean up
    df.drop(['upper_band_basic', 'lower_band_basic', 'upper_band', 'lower_band','supertrend'], axis=1, inplace=True) #, 'lower_band'
    return df

def numerical_features(df):
    original_columns = set(df.columns)
    df['ReturnSenti'] = np.log(df['SentiClose']).diff()
    df['ReturnIndex'] = np.log(df['IndexClose']).diff()
    # Lagged Returns of Index/Sentiment
    for lag in [1, 2, 3, 5, 8, 13]:
        df[f'Senti_Return_lag{lag}'] = df['ReturnSenti'].shift(lag)
        df[f'Index_Return_lag{lag}'] = df['ReturnIndex'].shift(lag)

    # Rolling Mean & StDev of Returns
    for window in [2, 3, 5, 8, 13, 21]:
        df[f'Rolling_Mean_Senti_Return_{window}'] = df['ReturnSenti'].rolling(window=window).mean()
        df[f'Rolling_StDev_Senti_Return_{window}'] = df['ReturnSenti'].rolling(window=window).std()
        df[f'Rolling_Mean_Index_Return_{window}'] = df['ReturnIndex'].rolling(window=window).mean()
        df[f'Rolling_StDev_Index_Return_{window}'] = df['ReturnIndex'].rolling(window=window).std()

    df['BodyRange'] = df['Close'] - df['Open']
    df['BarRange'] = df['High'] - df['Low']
    df['HL2'] = (df['High'] + df['Low']) / 2
    df['HLC3'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['OHLC4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['TradedValue'] = df['Volume'] * df['HLC3']
    df['CumulativeTradedValue'] = df['TradedValue'].cumsum()
    df['CumulativeVolume'] = df['Volume'].cumsum()
    df['VWAP'] = df['CumulativeTradedValue'] / df['CumulativeVolume']
    df['OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 -1)).cumsum()

    for window in [2, 3, 5, 8, 13, 21]:
        df[f'RollingClxRng_{window}'] = df['Close'] - df['Low'].rolling(window=window).min()
        df[f'RollingMaxRng_{window}'] = df['High'].rolling(window=window).max() - df['Low'].rolling(window=window).min()
        df[f'RollingOBV_{window}'] = df['OBV'].rolling(window=window).mean()   
        df['RollingTradedValue'] = df['TradedValue'].rolling(window=window).sum()
        
        # For CLOSE-PSAR Ocsilator 
        PSAR = tb.SAR(df['High'], df['Low'], acceleration=(0.0145+0.0236)/2, maximum=(0.145+0.236)/2)
        PSAROcsi = (df["Close"] - PSAR)
        df[f"PSAROcsiAvg_{window}"] = PSAROcsi.rolling(window=window).mean()
        df[f"PSAROcsiStd_{window}"] = PSAROcsi.rolling(window=window).std()
        df[f"PSARZScore_{window}"] = (PSAROcsi-df[f"PSAROcsiAvg_{window}"])/df[f"PSAROcsiStd_{window}"]

    # Momentum, Simple Moving Average, Exponential Moving Average
    for i in [2, 3, 5, 8, 13, 21]: # Adjust range depending on time windows needed
        df[f'MoM_{i}'] = df['Close'] - df['Close'].shift(i)
        df[f'SMA_{i}'] = df['Close'].rolling(i).mean()
        df[f'EMA_{i}'] = df['Close'].ewm(span=i).mean()
        df[f'CMO_{i}'] = tb.CMO(df['Close'], timeperiod=i)
        df[f'COG_{i}'] = (df['High'] - df['Low']) / 2
        df[f'MFI_{i}'] = tb.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=i)  
        df[f'RSI_{i}'] = tb.RSI(df['Close'], timeperiod=i)
        df[f'BUB_{i}'], df[f'BMB_{i}'], df[f'BLB_{i}'] = tb.BBANDS(df['Close'], timeperiod=i)
        df[f'KUC_{i}'], df[f'KMC_{i}'], df[f'KLC_{i}'] = tb.ATR(df['High'], df['Low'], df['Close'], timeperiod=i), df['Close'].rolling(i).mean(), tb.ATR(df['High'], df['Low'], df['Close'], timeperiod=i) - df['Close'].rolling(i).mean()
        df[f'MACD_{i}'], df[f'MACDSignal_{i}'], df[f'MACDHist_{i}'] = tb.MACD(df['Close'], fastperiod=i+3, slowperiod=(i+3)*2, signalperiod=i)
        df[f'SlowK_{i}'], df[f'SlowD_{i}'] = tb.STOCH(df['High'], df['Low'], df['Close'], fastk_period=(3+i)*2, slowk_period=3+i, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df[f'DCHigh_{i}'] = df['High'].rolling(i).max() # Donchian Upper band
        df[f'DCLow_{i}'] = df['Low'].rolling(i).min() # Donchian Lower band    
        df[f'DCMid_{i}'] = (df[f'DCHigh_{i}'] + df[f'DCLow_{i}']) / 2 # Donchian Middle band
        df[f'HL2SMA_{i}'] = df['HL2'].rolling(window=i).mean()
        df[f'SLOPE_{i}'] = tb.LINEARREG_SLOPE(df['Close'], timeperiod=i)
    
    df['ReturnClx'] = np.log(df['Close']).diff()
    df['ReturnHL2'] = np.log(df['HL2']).diff()

    # Lagged Returns
    for lag in [1, 2, 3, 5, 8, 13]:
        df[f'Clx_Return_lag{lag}'] = df['ReturnClx'].shift(lag)
        df[f'HL2_Return_lag{lag}'] = df['ReturnHL2'].shift(lag)
        
    # Rolling Mean & StDev of Returns
    for window in [2, 3, 5, 8, 13, 21]:
        df[f'Rolling_Mean_Clx_Return_{window}'] = df['ReturnClx'].rolling(window=window).mean()
        df[f'Rolling_StDev_Clx_Return_{window}'] = df['ReturnClx'].rolling(window=window).std()
        df[f'Rolling_Mean_HL2_Return_{window}'] = df['ReturnHL2'].rolling(window=window).mean()
        df[f'Rolling_StDev_HL2_Return_{window}'] = df['ReturnHL2'].rolling(window=window).std()

    df["LTSMA"] = df['Close'].rolling(33).mean()
    df['MAOcsi%'] = (df["Close"]/df["LTSMA"]-1)*100
    df["LTMAOcsiAvg"] = df["MAOcsi%"].rolling(3).mean()
    df["LTMAOcsiStd"] = df["MAOcsi%"].rolling(3).std()
    df["SMAZScore"] = (df["MAOcsi%"]-df["LTMAOcsiAvg"])/df["LTMAOcsiStd"] 
    df["TSF"] = tb.TSF(df['Close'], timeperiod=33)
    df['TSFOcsi%'] = (df["Close"]/df["TSF"]-1)*100
    df["TSFOcsiAvg"] = df["TSFOcsi%"].rolling(3).mean()
    df["TSFOcsiStd"] = df["TSFOcsi%"].rolling(3).std()
    df["TSFZScore"] = (df["TSFOcsi%"]-df["TSFOcsiAvg"])/df["TSFOcsiStd"] 
    numerical_features_list = list(set(df.columns) - original_columns)
    return df, numerical_features_list


def categorical_features(df):
    original_columns = set(df.columns)
    # Time-based features
    df['Year'] = df.index.year
    df['Quarter'] = df.index.quarter
    df['Month'] = df.index.month
    df['Week'] = df.index.isocalendar().week
    df['Date'] = df.index.day
    df['Day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6

    for window in [2, 3, 5, 8, 13, 21]:
        df[f"PSARbuyZone_{window}"] = df[f"PSARZScore_{window}"] > -2.33
        df[f"PSARprebuyZone_{window}"] = df[f"PSARZScore_{window}"].shift(1) <= -2.33
        df[f"PSARsellZone_{window}"] = df[f"PSARZScore_{window}"] < 2.33
        df[f"PSARpresellZone_{window}"] = df[f"PSARZScore_{window}"].shift(1) >= 2.33
        df[f"SMAOcsiBuySignal_{window}"] = np.where(df[f"PSARbuyZone_{window}"] & df[f"PSARprebuyZone_{window}"], 1, 0) 
        df[f"SMAOcsiSellSignal_{window}"] = np.where(df[f"PSARsellZone_{window}"] & df[f"PSARpresellZone_{window}"], -1, 0)

    df['RSI_signal'] = 0
    df.loc[df['RSI_13'] < 27, 'RSI_signal'] = 1
    df.loc[df['RSI_13'] > 73, 'RSI_signal'] = -1
    df['MFI_signal'] = 0
    df.loc[df['MFI_13'] < 27, 'MFI_signal'] = 1
    df.loc[df['MFI_13'] > 73, 'MFI_signal'] = -1

    buyZone = df["SMAZScore"] > -2.33
    prebuyZone = df["SMAZScore"].shift(1) <= -2.33
    sellZone = df["SMAZScore"] < 2.33
    presellZone = df["SMAZScore"].shift(1) >= 2.33
    df["SMAOcsiBuySignal"] = np.where(buyZone & prebuyZone, 1, 0) 
    df["SMAOcsiSellSignal"] = np.where(sellZone & presellZone, -1, 0)
    df['SMA_signal'] = 0
    df.loc[df["SMAOcsiBuySignal"] == 1, 'SMA_signal'] = 1
    df.loc[df['SMAOcsiSellSignal'] == 1, 'SMA_signal'] = -1

    buyZone = df["TSFZScore"] > -2.33
    prebuyZone = df["TSFZScore"].shift(1) <= -2.33
    sellZone = df["TSFZScore"] < 2.33
    presellZone = df["TSFZScore"].shift(1) >= 2.33
    df["TSFOcsiBuySignal"] = np.where(buyZone & prebuyZone, 1, 0) 
    df["TSFOcsiSellSignal"] = np.where(sellZone & presellZone, -1, 0)
    df['TSFArb_signal'] = 0
    df.loc[df["TSFOcsiBuySignal"] == 1, 'TSFArb_signal'] = 1
    df.loc[df['TSFOcsiSellSignal'] == 1, 'TSFArb_signal'] = -1
    # For EMA cross, 1 means golden cross and -1 means death cross
    df['EMA_signal'] = 0
    df['EMA_signal'] = np.where((df['EMA_8'] > df['EMA_21']), 1, df['EMA_signal'])
    df['EMA_signal'] = np.where((df['EMA_8'] < df['EMA_21']), -1, df['EMA_signal'])

    # Ensure that the signal changes only when there is a cross
    df['EMA_signal'] = df['EMA_signal'].diff()
    df['EMA_signal'] = np.where((df['EMA_signal'] != 0), df['EMA_signal'], 0)
    
    # Bollinger Bands signals
    df['BB_signal'] = 0
    df.loc[df['Close'] < df['BLB_13'], 'BB_signal'] = 1
    df.loc[df['Close'] > df['BUB_13'], 'BB_signal'] = -1

    # Keltner Channel signals
    df['KC_signal'] = 0
    df.loc[df['Close'] < df['KLC_13'], 'KC_signal'] = 1
    df.loc[df['Close'] > df['KUC_13'], 'KC_signal'] = -1

    # MACD signals
    df['MACD_signal'] = 0
    df.loc[df['MACD_8'] > df['MACDSignal_8'], 'MACD_signal'] = 1
    df.loc[df['MACD_8'] < df['MACDSignal_8'], 'MACD_signal'] = -1

    # Stochastic signals
    df['STOCH_signal'] = 0
    df.loc[(df['SlowK_13'] < 13) & (df['SlowD_13'] < 21), 'STOCH_signal'] = 1
    df.loc[(df['SlowK_13'] > 89) & (df['SlowD_13'] > 81), 'STOCH_signal'] = -1
    supertrend(df, period=13, multiplier=2.5)
    categorical_features_list = list(set(df.columns) - original_columns)
    return df, categorical_features_list

def prepare_data_for_ml(df, min_shift, max_shift):
    def compute_direction(df, shift_period):
        df['Return'] = df['Close'].shift(-shift_period) / df['Close'] - 1
        df['Direction'] = (df['Return'] > 0).astype(int)
        return df

    def check_balance(df):
        ones = sum(df['Direction'] == 1)
        zeros = sum(df['Direction'] == 0)
        total = len(df['Direction'].dropna())
        return abs((ones/total)*100 - (zeros/total)*100)

    def find_best_shift(df, min_shift, max_shift):
        smallest_difference = float('inf')
        best_shift = min_shift
        for shift_period in range(min_shift, max_shift + 1):
            temp_df = compute_direction(df.copy(), shift_period)
            difference = check_balance(temp_df)
            if difference < smallest_difference:
                smallest_difference = difference
                best_shift = shift_period
        return best_shift

    def rolling_window_check(df, stop_loss_level, take_profit_level):
        Direction = pd.DataFrame(index=df.index)
        Direction['Stop_or_Profit'] = 0  
        for i in range(len(df) - 1):
            current_price = df['Close'].iloc[i]
            for j in range(i + 1, len(df)):
                price_change = (df['Close'].iloc[j] - current_price) / current_price
                if price_change <= stop_loss_level:
                    Direction['Stop_or_Profit'].iloc[i] = 0  
                    break
                elif price_change > take_profit_level:
                    Direction['Stop_or_Profit'].iloc[i] = 1  
                    break
        df['Direction'] = Direction['Stop_or_Profit']
        return df

    df['Close_Change'] = df['Close'].pct_change()
    std_dev = df['Close_Change'].std()
    stop_loss_level = -0.125 * std_dev
    take_profit_level = 0.125 * std_dev

    best_shift = find_best_shift(df, min_shift, max_shift)
    df_shift = compute_direction(df.copy(), best_shift)
    balance_shift = check_balance(df_shift)

    df_window = rolling_window_check(df.copy(), stop_loss_level, take_profit_level)
    balance_window = check_balance(df_window)

    if balance_shift < balance_window:
        return df_shift, best_shift, 'find_best_shift', balance_shift
    else:
        return df_window, best_shift, 'rolling_window_check', balance_window

