import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

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

def training_values(data, start_year):
    # Store the minimum RMSE and corresponding training percentage for each year range
    best_training_percentages = []
    year_range = []

    # Loop over different year ranges
    for start_year in range(2024, start_year-1, -1):  # From 2024 down to 2004
        # Filter data for the year range (start_year to 2024)
        filtered_data = data[data['Date'].dt.year >= start_year]
        
        # Recalculate features and target for the filtered data
        X_filtered = filtered_data[['Open', 'High', 'Low', 'Volume']]
        y_filtered = filtered_data['Daily_Percentage_Change']
        
        # Store RMSE values for this iteration
        rmse_values = []
        training_percentages = []

        # Iterate over different training sizes
        for train_size in [i / 100 for i in range(40, 100)]:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, train_size=train_size, random_state=42)

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Lasso regression model
            lasso = Lasso(alpha=0.1, max_iter=10000, tol=1e-7)
            lasso.fit(X_train_scaled, y_train)

            # Predict and calculate RMSE
            y_pred = lasso.predict(X_test_scaled)
            rmse = root_mean_squared_error(y_test, y_pred)

            # Store RMSE and training percentage
            rmse_values.append(rmse)
            training_percentages.append(train_size * 100)  # Convert to percentage (e.g., 0.75 -> 75%)

        # Find the best training percentage corresponding to the minimum RMSE
        min_rmse_index = rmse_values.index(min(rmse_values))
        best_training_percent = training_percentages[min_rmse_index]

        best_training_percentages.append(best_training_percent)
        year_range.append(f"{start_year}-{2024}")

    # Print the best training percentage for each year range
    print("\nBest Training Percentages for Each Year Range:")
    for year, train_percent in zip(year_range, best_training_percentages):
        print(f"{year}: {train_percent}%")

    return year_range, best_training_percentages

def plot_results(year_range, best_training_percentages):
    # Plotting the best training percentages for each year range
    plt.figure(figsize=(10, 6))
    plt.plot(year_range, best_training_percentages, marker='o', color='red', label='Best Training Percentage')
    plt.title("Legjobb betanítási értékek 1975 és 2024 között")
    plt.xlabel("Year Range")
    plt.ylabel("Tanítási százalék (%)")
    plt.xticks(rotation=45)
    plt.ylim(20, 100)  # Since training percentages range from 20% to 100%
    plt.grid(True)
    plt.legend()
    plt.show()

def main(file_name, start_year=2015):
    data=load_data(file_name)
    year_range, min_mape_values = training_values(data, start_year=start_year)
    plot_results(year_range, min_mape_values)

main("data_50.csv",1975)