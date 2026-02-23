import os
import sqlite3
import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
from datetime import date, timedelta

def get_stock_prices(symbol, start_date, end_date):
    # Connect to SQLite database in project root
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "price.db"))
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    update_stock_prices(symbol, start_date, end_date, cur)

    # Fetch data from local database.
    cur.execute("""
        SELECT * FROM stock_prices WHERE symbol = ? AND date >= ? AND date < ? ORDER BY date; 
    """, (symbol, start_date, end_date))
    stock_prices = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    stock_prices = pd.DataFrame(stock_prices, columns = columns)

    stock_prices = stock_prices[stock_prices['volume'] != 0]
    stock_prices = stock_prices.reset_index(drop = True)
    stock_prices["date"] = pd.to_datetime(stock_prices["date"])

    conn.commit()
    cur.close()
    conn.close()

    return stock_prices

def update_stock_prices(symbol, start_date, end_date, cur):
    if start_date is not None and end_date is not None:
        if end_date > date.today().strftime("%Y-%m-%d"):
            end_date = date.today().strftime("%Y-%m-%d")

        # Get the oldest and latest stock prices' dates available in database.
        cur.execute("""
            SELECT MIN(date), MAX(date) FROM stock_prices WHERE symbol = ?;
        """, (symbol, ))
        min_date, max_date = cur.fetchone()
        # Exclusion adjustment
        if min_date is not None and max_date is not None:
            min_date = str(min_date)
            max_date = str(date.fromisoformat(str(max_date)) + timedelta(days = 1))

        # Fetch stock prices data from yfinance on dates that are not available in the database.
        ticker = yf.Ticker(symbol)
        data = None
        if min_date is None and max_date is None:
            data = ticker.history(start = start_date, end = end_date)
        elif start_date < min_date and end_date > max_date:
            data = ticker.history(start = start_date, end = end_date)
        elif start_date < min_date:
            data = ticker.history(start = start_date, end = min_date)
        elif end_date > max_date:
            data = ticker.history(start = max_date, end = end_date)

        # Update the database with the fetched data if any.
        if data is not None:
            stock_prices = []
            data = data.rename(columns = lambda x: x.replace(" ", "_"))
            for row in data.itertuples():
                stock_prices.append((symbol, str(row.Index.date()), row.Open, row.High, row.Low, row.Close, row.Volume, row.Dividends, row.Stock_Splits))

            cur.executemany("""
                INSERT OR IGNORE INTO stock_prices (symbol, date, open, high, low, close, volume, dividends, stock_splits)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, stock_prices)

def plot_stock_prices(stock_prices, split_date = None):
    fig, ax = plt.subplots(figsize = (12, 6))
    ax.plot(stock_prices["date"], stock_prices["close"], label = "Close price")

    if split_date is not None:
        split_dt = pd.to_datetime(split_date)
        ax.axvline(split_dt, color = "red", linestyle = "--", linewidth = 2, label = "Train/Test split")

    ax.set_title("Stock Closing Price")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha = 0.3)

    plt.show()

def split_train_test(stock_prices, split_date):
    split_dt = pd.to_datetime(split_date)
    train = stock_prices[stock_prices["date"] <= split_dt].reset_index(drop = True)
    test = stock_prices[stock_prices["date"] > split_dt].reset_index(drop = True)

    # Appends last 20 train rows to test for continuity.
    test = pd.concat([train.tail(20), test], axis = 0).reset_index(drop = True)
    
    return train, test