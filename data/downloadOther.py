import yfinance as yf

#symbol = "^GDAXI"  # For DAX P Index
symbol = "^FTSE"  # For DAX P Index

start_date = "1975-01-01"  # Start date
end_date = "2024-12-31"    # End date

data = yf.download(symbol, start=start_date, end=end_date)

#data.to_csv("dax_data_50.csv")
data.to_csv("ftse_data_50.csv")

print("DAX P 500 historical data saved to data.csv")