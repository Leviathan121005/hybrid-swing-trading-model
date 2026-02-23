import pandas as pd
import numpy as np

# For K-means clustering
def add_state_column(stock_prices):
    # Calculate OHLCV states
    ohlcv_states = pd.DataFrame(index = stock_prices.index)

    """
    Calculate each state component
    - O: Percentage change in open compared to previous day close
    - H: Percentage change in high compared to same day open
    - L: Percentage change in low compared to same day open
    - C: Percentage change in close compared to same day open
    - V: Percentage change in volume compared to previous day volume
    """
    ohlcv_states['O'] = ((stock_prices['open'] - stock_prices['close'].shift(1)) / stock_prices['close'].shift(1) * 100).round(4)
    ohlcv_states['H'] = ((stock_prices['high'] - stock_prices['open']) / stock_prices['open'] * 100).round(4)
    ohlcv_states['L'] = ((stock_prices['low'] - stock_prices['open']) / stock_prices['open'] * 100).round(4)
    ohlcv_states['C'] = ((stock_prices['close'] - stock_prices['open']) / stock_prices['open'] * 100).round(4)
    ohlcv_states['V'] = ((stock_prices['volume'] - stock_prices['volume'].shift(1)) / stock_prices['volume'].shift(1) * 100).round(4)
    
    stock_prices['state'] = ohlcv_states[['O', 'H', 'L', 'C', 'V']].dropna().apply(lambda row: [row['O'], row['H'], row['L'], row['C'], row['V']], axis = 1)
    stock_prices.dropna(inplace = True)

# For classifier
def get_technical_indicators(stock_prices, n_cci = 20, n_rsi = 14, n_pr = 14):
    # Compute CCI
    TP = (stock_prices['high'] + stock_prices['low'] + stock_prices['close']) / 3
    sma = TP.rolling(n_cci).mean()
    mean_dev = TP.rolling(n_cci).apply(lambda x: (x - x.mean()).abs().mean())
    cci = (TP - sma) / (0.015 * mean_dev)

    # Compute RSI
    delta = stock_prices['close'].diff()
    gain = delta.clip(lower = 0)
    loss = -delta.clip(upper = 0)
    avg_gain = gain.rolling(n_rsi).mean()
    avg_loss = loss.rolling(n_rsi).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Compute %R
    highest_high = stock_prices['high'].rolling(n_pr).max()
    lowest_low = stock_prices['low'].rolling(n_pr).min()
    pr = -100 * (highest_high - stock_prices['close']) / (highest_high - lowest_low)

    return cci, rsi, pr

def get_trade_actions(close_prices, peek = 7):
    # Determine trend based on MA15
    ma15 = close_prices.rolling(15).mean()
    ma15_diff5 = ma15.diff().rolling(5).sum()
    uptrend_idx = (close_prices > ma15) & (ma15_diff5 > 0)
    downtrend_idx = (close_prices < ma15) & (ma15_diff5 < 0)

    trend = np.zeros(len(close_prices)) 
    trend[uptrend_idx] = 1
    trend[downtrend_idx] = -1

    # Remove first 20 invalid values
    trend = trend[20:]

    # Generate trading signals based on future price range
    trading_signals = np.zeros(len(trend) - peek + 1)
    curr_post = 0

    # First trend index corresponds to 20th close prices index
    # Allow model to learn future patterns
    for t in range(20, len(trend) - peek + 1):
        window = close_prices[t:(t + peek)]
        x_min = window.min()
        x_max = window.max()
        if x_max == x_min:
            x_max += 1e-8

        # Days with no trend follows the last up or down trend
        if curr_post != trend[t - 20]:
            curr_post = trend[t - 20]
        
        if curr_post == 1:
            trading_signals[t - 20] = (close_prices[t] - x_min) / (x_max - x_min) * 0.5 + 0.5
        elif curr_post == -1:
            trading_signals[t - 20] = (close_prices[t] - x_min) / (x_max - x_min) * 0.5

    # Decide trading action based on the generated trading signals
    trade_actions = np.zeros(len(trading_signals), dtype = np.int8)
    trade_actions[0] = 2

    threshold = np.mean(trading_signals)
    prev_post = None
    curr_post = 1 if trading_signals[0] > threshold else -1

    for t in range(1, len(trading_signals)):
        prev_post = curr_post
        curr_post = 1 if trading_signals[t] > threshold else -1

        # Buy when the close price position today switches from low to high, and sell vice versa
        if prev_post == -1 and curr_post == 1:
            trade_actions[t] = 0
        elif prev_post == 1 and curr_post == -1:
            trade_actions[t] = 1
        else:
            trade_actions[t] = 2

    return trade_actions

# For comparison with MA trend driven trade actions
def get_ma_trend_actions(close_prices, lookback = 20):
    ma = close_prices.rolling(lookback).mean()
    actions = np.full(len(close_prices), 2)
    # Later in simulation, action decided on the closing price today will be used on the opening price tomorrow
    for t in range(lookback, len(close_prices) - 1):
        if close_prices[t] < ma[t]:
            actions[t] = 0
        elif close_prices[t] > ma[t]:
            actions[t] = 1
        else:
            actions[t] = 2
    return actions

# For comparison with break-out driven trade actions
def get_breakout_actions(close_prices, lookback = 20):
    rolling_high = close_prices.shift(1).rolling(lookback).max()
    rolling_low = close_prices.shift(1).rolling(lookback).min()

    actions = np.full(len(close_prices), 2)

    for t in range(lookback, len(close_prices) - 1):
        if close_prices[t] > rolling_high[t]:
            actions[t] = 0  # buy breakout
        elif close_prices[t] < rolling_low[t]:
            actions[t] = 1  # sell breakout
        else:
            actions[t] = 2

    return actions
