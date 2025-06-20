import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np  # Import numpy for absolute and division handling

def load_data(file_path):
    # Load the data
    path="../data/"+file_path
    column_names = ["Date", "Close", "High", "Low", "Open", "Volume"]
    data = pd.read_csv(path, skiprows=2, names=column_names)

    # Explicitly parse the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Check if there were any parsing issues (e.g., incorrect dates)
    if data['Date'].isnull().any():
        print("Warning: Some dates could not be parsed correctly.")

    # Compute Daily Percentage Change
    data['Daily_Percentage_Change'] = data['Close'].pct_change() * 100

    # Drop NaN Values
    data.dropna(inplace=True)

    return data

def calculate_mape(data, start_year):
    # Store the minimum MAPE for each year range
    min_mape_values = []
    year_range = []

    # Loop over different year ranges
    for start_year in range(2024, start_year-1, -1):  # From 2024 down to 2004
        # Filter data for the year range (start_year to 2024)
        filtered_data = data[data['Date'].dt.year >= start_year]
        
        # Recalculate features and target for the filtered data
        X_filtered = filtered_data[['Open', 'High', 'Low', 'Volume']]
        y_filtered = filtered_data['Daily_Percentage_Change']
        
        # Store MAPE values for this iteration
        mape_values = []

        # Iterate over different training sizes
        for train_size in [i / 100 for i in range(50, 100)]:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, train_size=train_size, random_state=42)

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Lasso regression model
            lasso = Lasso(alpha=0.1, max_iter=10000, tol=1e-7)
            lasso.fit(X_train_scaled, y_train)

            # Predict values
            y_pred = lasso.predict(X_test_scaled)

            # Compute MAPE (ignoring zero values in y_test to avoid division errors)
            mask = y_test != 0  # Avoid division by zero
            if np.any(mask):  # Check if there's at least one non-zero value
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) / 100
            else:
                mape = np.nan  # Assign NaN if all y_test values are zero

            # Store MAPE for this training size
            mape_values.append(mape)

        # Find the minimum MAPE for this year range and store it
        min_mape = np.nanmin(mape_values)  # Use nanmin to avoid NaN issues
        min_mape_values.append(min_mape)
        year_range.append(f"{start_year}-{2024}")

    # Print the minimum MAPE values for each year range
    print("Minimum MAPE values for each year range:")
    for year, mape in zip(year_range, min_mape_values):
        print(f"{year}: {mape:.5f}%")

    return year_range, min_mape_values

def plot_results(year_range, min_mape_values):
    # Plotting the minimum MAPE for each year range
    plt.figure(figsize=(10, 6))
    plt.plot(year_range, min_mape_values, marker='o', color='blue', label='Minimum MAPE')
    plt.title("Minimum MAPE érték 1995 és 2024 között")
    plt.xlabel("Year Range")
    plt.ylabel("Mean Absolute Percentage Error (%)")
    plt.xticks(rotation=45)
    plt.ylim(0, max(min_mape_values) + 0.02)  # Adjust y-axis based on max MAPE
    plt.grid(True)
    plt.legend()
    plt.show()

def main(file_name, start_year=2015):
    data=load_data(file_name)
    year_range, min_mape_values = calculate_mape(data, start_year=start_year)
    plot_results(year_range, min_mape_values)

main("ftse_data_50.csv", 1995)