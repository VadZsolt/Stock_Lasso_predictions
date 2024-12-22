import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler

# 1. Adatok beolvasása
def read_data(file_name):
    data = pd.read_csv(file_name)
    return data

# 2. Training funkció (adatok szétválasztása)
def training_function(data):
    # Csak az 'Adj Close' oszlopot használjuk
    train_features = data['Adj Close'].values[:-1].reshape(-1, 1)  # Az árfolyamokat használjuk jellemzőként
    train_stock_price = data['Adj Close'].values[1:]  # A célérték a következő nap árfolyama
    return train_features, train_stock_price

# 3. Testing funkció
def testing_function(data):
    test_features = data['Adj Close'].values[:-1].reshape(-1, 1)
    test_stock_price = data['Adj Close'].values[1:]
    return test_features, test_stock_price

# 4. LASSO regresszió tréningje
def LASSO_train(X, y, lambda_reg, eta=0.01, max_iter=1000):
    print("Input X shape:", X.shape)
    print("Input y shape:", y.shape)
    print("Lambda:", lambda_reg)
    print("Learning rate:", eta)
    N, d = X.shape
    w = np.zeros(d)  # Súlyok inicializálása
    best_w = w
    best_loss = float('inf')

    for iteration in range(max_iter):
        # Compute predictions
        predictions = X @ w
        
        # Compute MSE loss with L1 regularization
        mse_loss = np.mean((y - predictions) ** 2)
        l1_loss = lambda_reg * np.sum(np.abs(w))
        total_loss = mse_loss + l1_loss
        
        # Compute gradients
        grad_mse = -(2 / N) * X.T @ (y - predictions)  # MSE gradient
        grad_l1 = lambda_reg * np.sign(w)  # L1 regularization gradient
        
        # Update weights
        w_new = w - eta * (grad_mse + grad_l1)
        
        # Track the best weights
        if total_loss < best_loss:
            best_loss = total_loss
            best_w = w_new
        
        # Check for convergence
        if np.linalg.norm(w_new - w) < 1e-5:
            print(f"Konvergált {iteration + 1} iteráció után.")
            break
        
        w = w_new
    else:
        print("Figyelem! A súlyok nem konvergáltak a maximum iteráción belül.")
    
    return best_w

# 5. LASSO előrejelzés
def LASSO_predict(X, w):
    return X @ w

# 6. MAPE számítása
def calculate_mape(true_values, predictions):
    nonzero_mask = true_values != 0  # Nullák kezelése
    return np.mean(np.abs((true_values[nonzero_mask] - predictions[nonzero_mask]) / true_values[nonzero_mask]))

# 7. RMSE számítása
def calculate_rmse(true_values, predictions):
    return sqrt(np.mean((true_values - predictions) ** 2))/100

# Főprogram
def LASSO_REGRESSION(file_name, lambda_reg=0.1, eta=0.01, max_iter=1000):
    # 1. Adatok beolvasása
    data = read_data(file_name)

    print("Data columns:", data.columns)
    print("Data shape:", data.shape)
    print("First few rows:\n", data.head())

    # Ellenőrzés: vannak-e hiányzó vagy nem numerikus értékek
    if data.isnull().values.any():
        raise ValueError("Az adatfájl hiányzó értékeket tartalmaz. Ellenőrizd az adataidat!")

    # Adatok 80%-a tréningre, 20%-a tesztelésre
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    # 2-3. Adatok szétválasztása tréningre és tesztelésre
    train_features, train_stock_price = training_function(train_data)
    test_features, test_stock_price = testing_function(test_data)

    # Normalizálás (skálázás)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # 4. LASSO modell tréning
    model_weights = LASSO_train(train_features, train_stock_price, lambda_reg, eta, max_iter)

    # 5. Előrejelzés
    stock_price_predict = LASSO_predict(test_features, model_weights)

    # Ellenőrzés, hogy az előrejelzés és a tesztadatok száma megegyezik
    
    print(f"A tesztadatok hossza: {len(test_stock_price)}")
    print(f"Az előrejelzések hossza: {len(stock_price_predict)}")
    if len(stock_price_predict) != len(test_stock_price):
        min_len = min(len(test_stock_price), len(stock_price_predict))
        stock_price_predict = stock_price_predict[:min_len]  # Vágjuk le, ha szükséges

    # 6-7. MAPE és RMSE számítása
    mape = calculate_mape(test_stock_price, stock_price_predict)
    rmse = calculate_rmse(test_stock_price, stock_price_predict)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("Modell súlyai:", model_weights)

    # Grafikon rajzolása
    plot_predictions(test_data, test_stock_price, stock_price_predict)

# Grafikon létrehozása
def plot_predictions(test_data, true_values, predictions):
    # Create a copy of the test data to avoid modifying the original
    plot_data = test_data.copy()
    
    # Ensure the lengths match by using the shorter length
    min_len = min(len(plot_data), len(true_values), len(predictions))
    
    plot_data = plot_data.iloc[:min_len]
    true_values = true_values[:min_len]
    predictions = predictions[:min_len]

    # Add columns to the DataFrame
    plot_data['Predicted'] = predictions*1000
    print(predictions[0:10])
    plot_data['True'] = true_values
    print(true_values[0:10])
    plot_data['Date'] = pd.to_datetime(plot_data['Date'])

    plot_data = plot_data.sort_values(by='Date')

    # Grafikon készítése
    plt.figure(figsize=(12, 6))

    # Valódi és előrejelzett árak ábrázolása
    plt.plot(plot_data['Date'], plot_data['True'], label='Valódi Ár', color='blue', linestyle='--')
    plt.plot(plot_data['Date'], plot_data['Predicted'], label='Előrejelzett Ár', color='red', linestyle='-')

    # Címek és címkék
    plt.title("LASSO: Valódi és Előrejelzett Árak", fontsize=14)
    plt.xlabel("Dátum", fontsize=12)
    plt.ylabel("Ár", fontsize=12)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Program futtatása
file_name = "data1.csv"  # Az adatfájl neve
lambda_reg = 0.0001  # Regulárizációs paraméter
eta = 0.0001  # Tanulási sebesség
LASSO_REGRESSION(file_name, lambda_reg, eta)
