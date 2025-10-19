import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_PATH = "train.csv"
WAITING_TIME_MODEL_PATH = "waiting_time_model.pkl"

TARGET_COLUMN = "waiting_time_sec"

PREDICTIVE_FEATURES = [
    "distance_in_meters", 
    "duration_in_seconds", 
    "hour_sin",
    "hour_cos",
    "day_of_week",       
    "pickup_in_meters",
]

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Ошибка: Файл данных не найден по пути: {DATA_PATH}.")
    exit()

df['order_dt'] = pd.to_datetime(df['order_timestamp'])
df['tender_dt'] = pd.to_datetime(df['tender_timestamp'])
df[TARGET_COLUMN] = (df['tender_dt'] - df['order_dt']).dt.total_seconds().clip(lower=0)

df['hour_of_day'] = df['order_dt'].dt.hour
df['day_of_week'] = df['order_dt'].dt.dayofweek

df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

df.dropna(subset=PREDICTIVE_FEATURES + [TARGET_COLUMN], inplace=True)
print(f"Данные подготовлены. Размер: {df.shape}")

X = df[PREDICTIVE_FEATURES].values
y = df[TARGET_COLUMN].values

y_limit = np.quantile(y, 0.99)
y = np.clip(y, a_min=None, a_max=y_limit)

print("Начинаем обучение XGBRegressor...")

xgb_model = XGBRegressor(
    n_estimators=300,        # Увеличил количество деревьев, т.к. XGBoost работает быстрее
    learning_rate=0.05,      # Немного уменьшил шаг обучения для лучшей точности
    random_state=42, 
    n_jobs=-1,
    max_depth=6              # Разумная глубина для бустинга
)

xgb_model.fit(X, y)
print("Обучение завершено.")

y_pred = xgb_model.predict(X)
print(f"Средняя абсолютная ошибка (MAE) на тренировочных данных: {mean_absolute_error(y, y_pred):.2f} сек.")

joblib.dump(xgb_model, WAITING_TIME_MODEL_PATH)
print(f" Модель времени ожидания успешно сохранена в: {WAITING_TIME_MODEL_PATH}")

if __name__ == '__main__':
    test_order = {
        "distance_in_meters": 5000, 
        "duration_in_seconds": 900,  
        "pickup_in_meters": 200,    
        "order_timestamp": "2025-01-13 08:15:00"
    }
    
    test_dt = pd.to_datetime(test_order['order_timestamp'])
    test_order['hour_of_day'] = test_dt.hour
    test_order['day_of_week'] = test_dt.dayofweek
    test_order['hour_sin'] = np.sin(2 * np.pi * test_order['hour_of_day'] / 24)
    test_order['hour_cos'] = np.cos(2 * np.pi * test_order['hour_of_day'] / 24)
    
    test_input_list = [test_order[f] for f in PREDICTIVE_FEATURES]
    test_input_np = np.array(test_input_list).reshape(1, -1)
    
    predicted_waiting_time = xgb_model.predict(test_input_np)[0]
    
    print("\n--- Проверка прогноза времени ожидания ---")
    print(f"Прогнозируемое время ожидания до первого предложения: {predicted_waiting_time:.1f} секунд.")
    if predicted_waiting_time > 180:
        print("=> Высокое ожидание. Рынок напряжен.")
    else:
        print("=> Низкое ожидание. Рынок сбалансирован.")