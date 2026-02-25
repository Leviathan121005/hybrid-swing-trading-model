import numpy as np
import matplotlib.pyplot as plt
from hybrid_model import HybridModel
from technical_indicator_classifier import TechnicalIndicatorClassifier
from feature_extraction import get_ma_trend_actions, get_breakout_actions

def simulate_trade(stock_prices, model, plot = True):
    """
        In this simulation, the trade actions are done on the opening price from time 21 to T - 1.
        The trade actions for time 21 to T - 1 corresponds to the trade_actions on index 20 to T - 2.
        In other words, the trade actions decided today are based on information up to the day before.
    """
    if isinstance(model, HybridModel):
        classifier = model.classifier
        state_clusters = model.kmeans_predict(stock_prices)[20:-1]
        trade_actions = classifier.predict(stock_prices)
    elif isinstance(model, TechnicalIndicatorClassifier):
        # Indexing of 20 up to excluding last element is done in predict
        trade_actions = model.predict(stock_prices)
    elif model == "ma":
        trade_actions = get_ma_trend_actions(stock_prices["close"])[20:-1]
    elif model == "breakout":
        trade_actions = get_breakout_actions(stock_prices["close"])[20:-1]
    open_prices = stock_prices["open"].values[20:]

    position = 0
    entry_price = 0
    total_reward = 0

    total_reward_ts = np.zeros(len(open_prices) - 1)
    action_ts = np.zeros(len(open_prices) - 1)

    for t in range(1, len(open_prices)):
        price_t = open_prices[t]

        allowed_actions = [2]
        trade_action = trade_actions[t - 1]

        if isinstance(model, HybridModel):
            allowed_actions = [2]
            s_prev = state_clusters[t - 1]
            if position == 1 and trade_action == 1:
                allowed_actions.append(1)
            elif position == 0 and trade_action == 0:
                allowed_actions.append(0)
            action = model.get_action(s_prev, allowed_actions)
        else:
            action = trade_action
            
        # Stop loss
        if position == 1 and (price_t - entry_price) / entry_price <= -0.1:
            action = 1

        # Mark-to-market equivalent
        if t == len(open_prices) - 1:
            action = 1

        # Compute reward
        reward = 0
        if position == 0 and action == 0:
            position = 1
            entry_price = price_t
        elif position == 1 and action == 1:
            position = 0
            returns = (price_t - entry_price) / entry_price
            reward = returns - 0.002
        total_reward += reward

        total_reward_ts[t - 1] = total_reward
        action_ts[t - 1] = action 
    
    if plot:
        plot_trading_scheme(open_prices, total_reward_ts, action_ts)

    return total_reward


def plot_trading_scheme(open_prices, total_reward_ts, action_ts):    
    T = len(total_reward_ts)
    t = np.arange(T)

    returns = open_prices[1:] / open_prices[1] - 1

    fig, ax = plt.subplots(figsize = (14, 6))

    # Plot both series on same y-axis
    for i in range(len(returns) - 1):
        if action_ts[i] == 0:
            color = "green"
        elif action_ts[i] == 1:
            color = "red"
        else:                    
            color = "gray"
        
        ax.plot(
            t[i:(i + 2)],
            returns[i:(i + 2)],
            color = color,
            linewidth = 1.5,
            alpha = 0.7,
            label = 'Buy & Hold' if i == 0 else ""
        )
        
        ax.plot(
            t[i:(i + 2)],
            total_reward_ts[i:(i + 2)],
            color = color,
            linewidth = 3,
            linestyle = '--',
            alpha = 0.9,
            label = 'Cumulative Returns' if i == 0 else ""
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Return")
    ax.set_title("Cumulative Returns vs Buy & Hold")
    ax.grid(alpha = 0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color = 'black', lw = 3, label = 'Cumulative Returns (thick)'),
        Line2D([0], [0], color = 'black', lw = 1.5, linestyle = '--', label = 'Buy & Hold (thin)'),
        Line2D([0], [0], color = 'green', lw = 2, label = 'BUY'),
        Line2D([0], [0], color = 'red', lw = 2, label = 'SELL'),
        Line2D([0], [0], color = 'gray', lw = 2, label = 'HOLD')
    ]
    ax.legend(handles = legend_elements, loc = 'upper left')

    plt.tight_layout()
    plt.show()