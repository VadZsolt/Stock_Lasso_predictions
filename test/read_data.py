import pandas as pd
import numpy as np


def read_data(file_path):
    column_names = ["Date", "Adj_Close", "Close", "High", "Low", "Open", "Volume"]
    data = pd.read_csv(file_path, skiprows=2, names=column_names, parse_dates=["Date"])
    data.dropna(inplace=True)
    X = data[['Open', 'High', 'Low', 'Volume']].values
    y = data['Close'].values

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std
    return X_scaled, y