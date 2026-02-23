import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from feature_extraction import get_technical_indicators, get_trade_actions

class TechnicalIndicatorClassifier:
    def __init__(self, model = "GaussianNB", peek = 3):
        self.scaler = StandardScaler()
        if model == "GaussianNB":
            self.classifier = GaussianNB(priors = [1/3, 1/3, 1/3])
        self.peek = peek

    def train(self, stock_prices):
        cci, rsi, pr = get_technical_indicators(stock_prices)

        X = np.column_stack([cci, rsi, pr])[20:-(self.peek - 1)]
        X_scaled = self.scaler.fit_transform(X)
        y = get_trade_actions(stock_prices["close"], self.peek)
        
        # Map technical indicator values of the closing price on the day before to the rewarding trade action for today generated from peeking at future values.
        self.classifier.fit(X_scaled[:-1], y[1:])
    
    def predict(self, stock_prices):
        cci, rsi, pr = get_technical_indicators(stock_prices)

        X = np.column_stack([cci, rsi, pr])[20:]
        X_scaled = self.scaler.transform(X)
        
        return self.classifier.predict(X_scaled[:-1])