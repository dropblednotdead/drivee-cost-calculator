import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
import numpy as np

DATA_PATH = "train.csv" 
BASE_MODEL_PATH = "base_model.pkl"

BASE_MODEL_FEATURES = ["distance_in_meters", "duration_in_seconds"] 
TARGET_COLUMN = "price_start_local"

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Ошибка: Файл данных не найден по пути: {DATA_PATH}. Пожалуйста, обновите DATA_PATH.")
    exit()

df.dropna(subset=BASE_MODEL_FEATURES + [TARGET_COLUMN], inplace=True)
df = df[(df[TARGET_COLUMN] > 50) & (df[TARGET_COLUMN] < df[TARGET_COLUMN].quantile(0.99))]

X = df[BASE_MODEL_FEATURES].values
y = df[TARGET_COLUMN].values

print("Начинаем обучение простой линейной регрессии...")
base_model = LinearRegression()
base_model.fit(X, y)

print("Обучение завершено.")
print(f"Модель предсказывает цену по формуле:\nЦена = ({base_model.coef_[0]:.4f} * distance) + ({base_model.coef_[1]:.4f} * duration) + ({base_model.intercept_:.4f})")

joblib.dump(base_model, BASE_MODEL_PATH)
print(f"Базовая модель успешно сохранена в: {BASE_MODEL_PATH}")

if __name__ == '__main__':
    test_order_long = {
        "distance_in_meters": 10000,
        "duration_in_seconds": 1200, 
        "price_start_local": 200,    
    }
    
    test_order_short = {
        "distance_in_meters": 2000, 
        "duration_in_seconds": 300,  
        "price_start_local": 100,    
    }
    
    print("\n--- Проверка прогноза базовой цены ---")
    
    for name, order in {"Длинная поездка": test_order_long, "Короткая поездка": test_order_short}.items():
        
        test_input_list = [order[f] for f in BASE_MODEL_FEATURES]
        
        test_input_np = np.array(test_input_list).reshape(1, -1)
        
        predicted_base_price = base_model.predict(test_input_np)[0]
        
        price_anomaly = (predicted_base_price - order["price_start_local"]) / predicted_base_price
        
        print(f"\n[{name}]")
        print(f"   Факторы: Расстояние={order['distance_in_meters']/1000:.1f}км, Время={order['duration_in_seconds']/60:.0f}мин.")
        print(f"   Базовая 'Честная' Цена (прогноз): {predicted_base_price:.2f} руб.")
        print(f"   Текущая Цена (ввод): {order['price_start_local']} руб.")
        print(f"   Создаваемый признак Price_Anomaly: {price_anomaly:.3f}")

        if price_anomaly > 0.2:
             print(f"   => Текущая цена слишком низкая! (Скидка > 20%)")
        elif price_anomaly < -0.2:
             print(f"   => Текущая цена слишком высокая! (Наценка > 20%)")
        else:
             print(f"   => Текущая цена адекватна.")