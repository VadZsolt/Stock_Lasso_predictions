import pandas as pd

# Az eredeti CSV fájl betöltése
file_name = "data.csv"  # Helyettesítsd az eredeti fájl nevével
data = pd.read_csv(file_name)

# Válasszuk ki a szükséges oszlopokat
selected_columns = ['Date', 'Adj Close']
cleaned_data = data[selected_columns].dropna()  # Távolítsuk el a hiányzó adatokat

# Mentés új CSV fájlként, amelyet majd a modell használ
cleaned_data.to_csv("../own/data1.csv", index=False)
print("Az adatfájl elkészült: data1.csv")