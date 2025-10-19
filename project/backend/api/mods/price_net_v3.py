import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.quantization
import joblib
import numpy as np
import time
import os
from django.conf import settings


# ----------------------------------------------------
# 1. ТОЧНАЯ АРХИТЕКТУРА ИЗ РАБОЧЕГО КОДА
# ----------------------------------------------------

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

        # SHARED BODY с RESIDUAL BLOCKS
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

        # УГЛУБЛЕННЫЕ ГОЛОВЫ
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


# ----------------------------------------------------
# 2. КОНСТАНТЫ ИЗ РАБОЧЕГО КОДА
# ----------------------------------------------------

# Важен порядок! Должен совпадать с порядком при обучении.
PRICE_NET_FEATURES = [
    "price_start_local", "distance_in_meters", "duration_in_seconds",
    "pickup_in_meters", "pickup_in_seconds", "driver_rating",
    "waiting_time_sec", "hour_sin", "hour_cos",
    "day_of_week", "price_anomaly", "user_rating"  # ← 12 ПРИЗНАКОВ!
]

WAITING_TIME_MODEL_FEATURES = [
    "distance_in_meters", "duration_in_seconds", "hour_sin",
    "hour_cos", "day_of_week", "pickup_in_meters",
]

BASE_MODEL_FEATURES = ["distance_in_meters", "duration_in_seconds"]


# ----------------------------------------------------
# 3. КЛАСС ПРОГНОЗИРОВАНИЯ (С ПРАВИЛЬНЫМИ ПУТЯМИ)
# ----------------------------------------------------

class PricePredictor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = None
        self.scaler = None
        self.waiting_time_model = None
        self.base_model = None

        # ИСПОЛЬЗУЕМ ТОЧНЫЙ СПИСОК ПРИЗНАКОВ ИЗ РАБОЧЕГО КОДА
        self.features = PRICE_NET_FEATURES  # 12 признаков!
        self.wt_features = WAITING_TIME_MODEL_FEATURES
        self.base_features = BASE_MODEL_FEATURES

        self._load_all_models()

    def _get_path(self, filename):
        return os.path.join(self.base_dir, filename)

    def _load_all_models(self):
        """Загружает все модели и масштабировщик"""
        try:
            # --- Загрузка PriceNet и Scaler ---
            MODEL_PATH = self._get_path('pricenet_model.pth')
            SCALER_PATH = self._get_path('scaler.pkl')

            print(f"🔄 Загрузка моделей из: {self.base_dir}")
            print(f"🔍 Поиск модели: {MODEL_PATH}")
            print(f"🔍 Поиск scaler: {SCALER_PATH}")

            # Проверяем существование файлов
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Файл scaler не найден: {SCALER_PATH}")

            # СОЗДАЕМ МОДЕЛЬ С ПРАВИЛЬНЫМ КОЛИЧЕСТВОМ ПРИЗНАКОВ
            input_dim = len(self.features)  # 12 признаков!
            model_fp32 = PriceNet(input_dim=input_dim)

            # Загружаем веса с обработкой variance параметров
            state_dict = torch.load(MODEL_PATH, map_location='cpu')

            # Убираем параметры variance если они мешают
            keys_to_remove = []
            for key in state_dict.keys():
                if 'variance' in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del state_dict[key]
                print(f"⚠Удален параметр: {key}")

            # Загружаем state_dict (strict=False для игнорирования лишних параметров)
            model_fp32.load_state_dict(state_dict, strict=False)
            model_fp32.eval()

            self.model = torch.quantization.quantize_dynamic(
                model_fp32, {nn.Linear}, dtype=torch.qint8
            )
            print(f"PriceNet загружена. Признаков: {input_dim}")

            # Загружаем scaler
            self.scaler = joblib.load(SCALER_PATH)
            print(f"Масштабировщик загружен.")

            # --- Загрузка вспомогательных моделей ---
            WAITING_TIME_MODEL_PATH = self._get_path('waiting_time_model.pkl')
            BASE_MODEL_PATH = self._get_path('base_model.pkl')

            if os.path.exists(WAITING_TIME_MODEL_PATH):
                self.waiting_time_model = joblib.load(WAITING_TIME_MODEL_PATH)
                print(f"Модель времени ожидания загружена.")
            else:
                print(f"Модель времени ожидания не найдена")

            if os.path.exists(BASE_MODEL_PATH):
                self.base_model = joblib.load(BASE_MODEL_PATH)
                print(f"Базовая модель загружена.")
            else:
                print(f"Базовая модель не найдена")

            print("Все модели успешно загружены!")

        except Exception as e:
            print(f"Ошибка загрузки моделей: {e}")
            raise

    def _engineer_features(self, order_data):
        """Выполняет инженерию признаков (12 признаков)"""
        # 1. Генерация временных признаков
        order_dt = pd.to_datetime(order_data.get('order_timestamp', pd.Timestamp.now()))

        order_data['day_of_week'] = order_dt.dayofweek
        order_data['hour_of_day'] = order_dt.hour
        order_data['hour_sin'] = np.sin(2 * np.pi * order_data['hour_of_day'] / 24)
        order_data['hour_cos'] = np.cos(2 * np.pi * order_data['hour_of_day'] / 24)

        # 2. Добавляем user_rating (значение по умолчанию если нет)
        if 'user_rating' not in order_data:
            order_data['user_rating'] = 4.5  # средний рейтинг по умолчанию

        # 3. Прогноз времени ожидания (если модель есть)
        if self.waiting_time_model is not None:
            time_input_list = [order_data[f] for f in self.wt_features]
            time_sample_np = np.array(time_input_list, dtype=np.float32).reshape(1, -1)
            predicted_waiting_time = self.waiting_time_model.predict(time_sample_np)[0]
            order_data["waiting_time_sec"] = predicted_waiting_time
        else:
            order_data["waiting_time_sec"] = 300  # значение по умолчанию

        # 4. Прогноз аномалии цены (если модель есть)
        if self.base_model is not None:
            base_price_input_list = [order_data[f] for f in self.base_features]
            base_price_sample_np = np.array(base_price_input_list, dtype=np.float32).reshape(1, -1)
            predicted_base_price = self.base_model.predict(base_price_sample_np)[0]
        else:
            predicted_base_price = order_data["price_start_local"] * 1.2

        # Расчет price_anomaly
        price_anomaly = (predicted_base_price - order_data["price_start_local"]) / predicted_base_price
        order_data["price_anomaly"] = price_anomaly

        # 5. Сборка финального вектора (12 признаков!)
        input_list = [order_data[f] for f in self.features]
        sample_np = np.array(input_list, dtype=np.float32).reshape(1, -1)

        return sample_np, order_data["waiting_time_sec"], predicted_base_price

    def get_recommendations(self, order_data):
        """Основной метод прогноза"""
        start_time = time.time()

        try:
            # 1. Инженерия признаков
            sample_np, predicted_waiting_time, predicted_base_price = self._engineer_features(order_data)

            # 2. Препроцессинг для PriceNet
            sample_scaled = self.scaler.transform(sample_np)
            sample = torch.tensor(sample_scaled, dtype=torch.float32)

            # 3. Предсказание PriceNet
            with torch.inference_mode():
                price_rel_pred, prob_raw_pred = self.model(sample)

            prob_pred = torch.sigmoid(prob_raw_pred)
            price_pred_real = price_rel_pred * order_data["price_start_local"]

            # 4. Формирование рекомендаций
            sorted_indices = torch.argsort(price_pred_real[0])
            recommendations = []

            info = {
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "predicted_waiting_time_sec": round(predicted_waiting_time, 1),
                "predicted_base_price_rub": round(predicted_base_price, 2),
                "price_anomaly": round(order_data["price_anomaly"], 3),
            }

            for i in sorted_indices:
                price = price_pred_real[0, i].item()
                prob = prob_pred[0, i].item()

                recommendations.append({
                    "price": round(price),
                    "probability": round(prob, 3),
                    "expected_revenue": round(price * prob, 2)
                })

            return recommendations, info

        except Exception as e:
            print(f"Ошибка в get_recommendations: {e}")
            raise


# Создаем инстанс predictor с обработкой ошибок
try:
    predictor = PricePredictor()
    print("✅ PricePredictor успешно инициализирован")
except Exception as e:
    print(f"Ошибка инициализации PricePredictor: {e}")
    predictor = None


predictor = PricePredictor()