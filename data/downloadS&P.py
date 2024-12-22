import yfinance as yf

# Define the S&P 500 symbol (SPY is an ETF tracking the S&P 500 index)
symbol = "^GSPC"  # For S&P 500 Index

start_date = "2020-11-30"  # Start date
end_date = "2024-11-30"    # End date

data = yf.download(symbol, start=start_date, end=end_date)

data.to_csv("data.csv")

print("S&P 500 historical data saved to data.csv")