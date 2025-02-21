import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# Adatok betöltése
column_names = ["Date","Adj Close", "Close", "High", "Low", "Open", "Volume"]
data = pd.read_csv("../data/data.csv", skiprows=2, names=column_names, parse_dates=["Date"])
# Napi százalékos változás számítása
data['Daily_Percentage_Change'] = data['Close'].pct_change() * 100

# Null értékek eltávolítása
data.dropna(inplace=True)

# A jellemzők és a célértékek meghatározása
X = data[['Open', 'High', 'Low', 'Volume']]  # Jellemzők
y = data['Daily_Percentage_Change']  # Célérték: napi százalékos változás

# RMSE értékek tárolásához
train_sizes = []
rmse_values = []

# Train-test split méretének iterálása
for train_size in [i / 100 for i in range(50, 100)]:
    # Adatok felosztása
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Jellemzők skálázása
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Lasso regresszió betanítása
    lasso = Lasso(alpha=0.1, max_iter=10000, tol=1e-7)
    lasso.fit(X_train_scaled, y_train)

    # Előrejelzés és RMSE számítása
    y_pred = lasso.predict(X_test_scaled)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Eredmények tárolása
    train_sizes.append(train_size)
    rmse_values.append(rmse)

# RMSE értékek ábrázolása a train_size függvényében

rmse_values_converted = [float(value) for value in rmse_values]

print("Négyzetes középhiba értékek:")
for value in rmse_values_converted:
    print(value)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, rmse_values, marker='o', color='blue', label='Négyzetes középhiba')
plt.title("Négyzetes középhiba értékek a train_size függvényében")
plt.xlabel("Tanítasi százalék")
plt.ylabel("Négyzetes középhiba százalékok")
plt.grid(True)
plt.legend()
plt.show()
