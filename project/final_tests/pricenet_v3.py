import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.quantization
import joblib 
import numpy as np
import time

# --- 1. АРХИТЕКТУРНЫЕ КЛАССЫ  ---

class ResidualBlock(nn.Module):
    """Блок с остаточной связью (ResNet)"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim) 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = self.relu(self.linear(x))
        return residual + x 

class PriceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_recommendations=3):
        super().__init__()
        
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim), 
            ResidualBlock(hidden_dim), 
        )

        # УГЛУБЛЕННЫЕ ГОЛОВЫ (Tuning Heads)
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, num_recommendations)
        )
        
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(hidden_dim // 2, num_recommendations)
        )

        # Адаптивные веса потерь
        self.log_price_variance = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_prob_variance = nn.Parameter(torch.zeros(1), requires_grad=True)
        
    def forward(self, x):
        shared_out = self.shared_body(x)
        price_raw = self.price_head(shared_out)
        price = 0.8 + 0.7 * torch.sigmoid(price_raw) 
        prob_raw = self.prob_head(shared_out)
        return price, prob_raw

# --- 2. КОНСТАНТЫ ПУТЕЙ И ПРИЗНАКОВ ---

MODEL_PATH = "pricenet_model.pth"
SCALER_PATH = "scaler.pkl"
WAITING_TIME_MODEL_PATH = "waiting_time_model.pkl"
BASE_MODEL_PATH = "base_model.pkl"

BASE_MODEL_FEATURES = ["distance_in_meters", "duration_in_seconds"]

WAITING_TIME_MODEL_FEATURES = [
    "distance_in_meters", "duration_in_seconds", "hour_sin",
    "hour_cos", "day_of_week", "pickup_in_meters",
]


PRICE_NET_FEATURES = [
    "price_start_local", "distance_in_meters", "duration_in_seconds",
    "pickup_in_meters", "pickup_in_seconds", "driver_rating",
    "waiting_time_sec", "hour_sin", "hour_cos",
    "day_of_week", "price_anomaly" 
]

# --- 3. КЛАСС ПРЕДИКТОР  ---

class PricePredictor:
    def __init__(self, input_dim):
        self._load_all_models(input_dim)
        self.features = PRICE_NET_FEATURES
        self.wt_features = WAITING_TIME_MODEL_FEATURES
        self.base_features = BASE_MODEL_FEATURES

    def _load_all_models(self, input_dim):
        print("--- Загрузка моделей (Холодный старт) ---")
        
        model_fp32 = PriceNet(input_dim=input_dim)
        try:
            model_fp32.load_state_dict(torch.load(MODEL_PATH))
            model_fp32.eval()
        except Exception as e:
            raise FileNotFoundError(f"Ошибка загрузки PriceNet: {e}")

        self.model = torch.quantization.quantize_dynamic(
            model_fp32, {nn.Linear}, dtype=torch.qint8 
        )
        print("✔ PriceNet (квантованная) загружена.")

        try:
            self.scaler = joblib.load(SCALER_PATH)
            self.waiting_time_model = joblib.load(WAITING_TIME_MODEL_PATH)
            self.base_model = joblib.load(BASE_MODEL_PATH)
            print("✔ Все вспомогательные модели (Scaler, XGBoost, LR) загружены.")
        except Exception as e:
            raise FileNotFoundError(f"Ошибка загрузки вспомогательных моделей: {e}")


    def predict(self, order_data: dict):
        start_time = time.time()
        
        # 1. Инженерия времени (часы, синусы/косинусы)
        order_dt = pd.to_datetime(order_data['order_timestamp'])
        order_data['day_of_week'] = order_dt.dayofweek
        order_data['hour_of_day'] = order_dt.hour
        order_data['hour_sin'] = np.sin(2 * np.pi * order_data['hour_of_day'] / 24)
        order_data['hour_cos'] = np.cos(2 * np.pi * order_data['hour_of_day'] / 24)
        
        # 2. Прогноз Времени Ожидания (XGBoost)
        time_input = np.array([order_data[f] for f in self.wt_features], dtype=np.float32).reshape(1, -1)
        predicted_waiting_time = self.waiting_time_model.predict(time_input)[0]
        order_data["waiting_time_sec"] = predicted_waiting_time
        
        # 3. Прогноз Базовой Цены (LR) и Аномалии
        base_price_input = np.array([order_data[f] for f in self.base_features], dtype=np.float32).reshape(1, -1)
        predicted_base_price = self.base_model.predict(base_price_input)[0]
        price_anomaly = (predicted_base_price - order_data["price_start_local"]) / predicted_base_price
        order_data["price_anomaly"] = price_anomaly

        # 4. Подготовка входных данных для PriceNet
        sample_np = np.array([order_data[f] for f in self.features], dtype=np.float32).reshape(1, -1)
        sample_scaled = self.scaler.transform(sample_np)
        sample = torch.tensor(sample_scaled, dtype=torch.float32)

        # 5. Прогноз PriceNet (Инференс)
        with torch.inference_mode():
            price_rel_pred, prob_raw_pred = self.model(sample)

        prob_pred = torch.sigmoid(prob_raw_pred)
        price_pred_real = price_rel_pred * order_data["price_start_local"]
        
        # 6. Форматирование результата
        result = {
            'waiting_time_sec': predicted_waiting_time,
            'price_anomaly': price_anomaly,
            'predictions': []
        }
        
        # Сортировка по цене (для удобства вывода)
        sorted_indices = torch.argsort(price_pred_real[0]) 
        for i in sorted_indices:
            result['predictions'].append({
                'price': price_pred_real[0,i].item(),
                'probability': prob_pred[0,i].item()
            })
            
        end_time = time.time()
        print(f"\n✔ Прогноз выполнен за: {end_time - start_time:.4f} секунд.")
        return result

# --- 4. ЗАПУСК ИНФЕРЕНСА ---

if __name__ == '__main__':
    # Пример данных для прогноза
    test_order = {
        "order_timestamp": "2025-01-13 13:15:00",
        "price_start_local": 150,
        "distance_in_meters": 4000,
        "duration_in_seconds": 3600,
        "pickup_in_meters": 800,
        "pickup_in_seconds": 120,
        "driver_rating": 4.7,
    }

    predictor = PricePredictor(input_dim=len(PRICE_NET_FEATURES))
    
    prediction_result = predictor.predict(test_order)
    
    print("\n--- Прогноз для нового заказа ---")
    print(f"Прогноз времени ожидания: {prediction_result['waiting_time_sec']:.1f} сек.")
    print(f"Признак Price_Anomaly: {prediction_result['price_anomaly']:.3f}")
    print("-" * 40)
    
    for p in prediction_result['predictions']:
        print(f"Цена: {p['price']:.0f} руб -> Вероятность согласия: {p['probability']*100:.1f}%")