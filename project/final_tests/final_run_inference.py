import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.quantization
import joblib 
import numpy as np
import time
from typing import Dict, Any, List

# ==============================================================================
# 1. АРХИТЕКТУРНЫЕ КЛАССЫ
# ==============================================================================

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
        self.log_price_variance = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_prob_variance = nn.Parameter(torch.zeros(1), requires_grad=True)
        
    def forward(self, x):
        shared_out = self.shared_body(x)
        price_raw = self.price_head(shared_out)
        price = 0.8 + 0.7 * torch.sigmoid(price_raw) 
        prob_raw = self.prob_head(shared_out)
        return price, prob_raw

# ==============================================================================
# 2. КОНСТАНТЫ ПУТЕЙ И ПРИЗНАКОВ (ОСТАЮТСЯ БЕЗ ИЗМЕНЕНИЙ)
# ==============================================================================

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

# ==============================================================================
# 3. КЛАСС ПРЕДИКТОР (МОДИФИЦИРОВАН)
# ==============================================================================

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
            self.model = torch.quantization.quantize_dynamic(
                model_fp32, {nn.Linear}, dtype=torch.qint8 
            )
            print("✔ PriceNet (квантованная) загружена.")
            self.scaler = joblib.load(SCALER_PATH)
            self.waiting_time_model = joblib.load(WAITING_TIME_MODEL_PATH)
            self.base_model = joblib.load(BASE_MODEL_PATH)
            print("✔ Все вспомогательные модели загружены.")
        except Exception as e:
            raise FileNotFoundError(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить файлы моделей. Проверьте пути: {e}")

    
    # --- МОДИФИЦИРОВАННЫЙ МЕТОД: ОСНОВНОЙ ПАЙПЛАЙН ПРОГНОЗА ---
    def predict(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет инференс для одного заказа."""
        
        order_dt = pd.to_datetime(order_data['order_timestamp'])
        data = order_data.copy()
        
        data['day_of_week'] = order_dt.dayofweek
        data['hour_of_day'] = order_dt.hour
        data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)
        
        time_input = np.array([data[f] for f in self.wt_features], dtype=np.float32).reshape(1, -1)
        predicted_waiting_time = self.waiting_time_model.predict(time_input)[0]
        data["waiting_time_sec"] = predicted_waiting_time
        
        base_price_input = np.array([data[f] for f in self.base_features], dtype=np.float32).reshape(1, -1)
        predicted_base_price = self.base_model.predict(base_price_input)[0]
        price_anomaly = (predicted_base_price - data["price_start_local"]) / predicted_base_price
        data["price_anomaly"] = price_anomaly

        sample_np = np.array([data[f] for f in self.features], dtype=np.float32).reshape(1, -1)
        sample_scaled = self.scaler.transform(sample_np)
        sample = torch.tensor(sample_scaled, dtype=torch.float32)

        with torch.inference_mode():
            price_rel_pred, prob_raw_pred = self.model(sample)

        prob_pred = torch.sigmoid(prob_raw_pred)
        price_pred_real = price_rel_pred * data["price_start_local"]
        
        predictions = []
        sorted_indices = torch.argsort(price_pred_real[0]) 
        
        for i in sorted_indices:
            predictions.append({
                'price': price_pred_real[0,i].item(),
                'probability': prob_pred[0,i].item()
            })
            
        return predictions, predicted_waiting_time, price_anomaly
    
    def process_dataframe(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Обрабатывает весь DataFrame и возвращает DataFrame с результатами."""
        
        results: List[Dict[str, Any]] = []
        total_orders = len(test_df)
        start_time = time.time()
        
        print(f"\n--- Начинается пакетный инференс для {total_orders} заказов ---")

        for index, row in test_df.iterrows():

            order_data = row.to_dict()
            
            predictions, wt_sec, anomaly = self.predict(order_data)
            
            max_er = 0.0
            
            result_row = {'order_id': order_data['order_id']}
            
            for i, p in enumerate(predictions):
                result_row[f'price_{i+1}'] = round(p['price'], 2)
                result_row[f'prob_{i+1}'] = round(p['probability'], 4)
                
                er = p['price'] * p['probability']
                if er > max_er:
                    max_er = er
            
            result_row['expected_revenue_max'] = round(max_er, 2)
            
            results.append(result_row)
            
            if (index + 1) % 1000 == 0:
                print(f"Обработано {index + 1}/{total_orders} заказов...")

        end_time = time.time()
        
        print(f"--- Пакетный инференс завершен. Время: {end_time - start_time:.2f} сек. ---")
        return pd.DataFrame(results)

# ==============================================================================
# 4. ФИНАЛЬНЫЙ ЗАПУСК И СОХРАНЕНИЕ
# ==============================================================================

if __name__ == '__main__':
    TEST_FILE = "test.csv"
    OUTPUT_FILE = "predictions.csv"
    
    try:
        test_df = pd.read_csv(TEST_FILE)
        if 'order_id' not in test_df.columns:
             test_df['order_id'] = range(len(test_df))
        print(f"✔ Загружен файл: {TEST_FILE} ({len(test_df)} заказов)")
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {TEST_FILE} не найден. Убедитесь, что он в корневой папке.")
        exit()

    try:
        predictor = PricePredictor(input_dim=len(PRICE_NET_FEATURES))
    except FileNotFoundError as e:
        print(f"Критическая ошибка инициализации: {e}")
        exit()

    final_predictions_df = predictor.process_dataframe(test_df)

    final_predictions_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*60)
    print(f"Результаты сохранены в файл: {OUTPUT_FILE}")
    print("="*60)
    print("Пример первых 5 строк результата:")
    print(final_predictions_df.head())