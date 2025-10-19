import torch
import torch.nn as nn
import torch.quantization
import joblib 
import numpy as np
import time


class PriceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_recommendations=3):
        super().__init__()
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.price_head = nn.Linear(hidden_dim, num_recommendations)
        self.prob_head = nn.Linear(hidden_dim, num_recommendations)

    def forward(self, x):
        shared_out = self.shared_body(x)
        price_raw = self.price_head(shared_out)
        price = 0.8 + 0.7 * torch.sigmoid(price_raw) 
        prob_raw = self.prob_head(shared_out)
        return price, prob_raw

start_time = time.time()

MODEL_PATH = "pricenet_model.pth"
SCALER_PATH = "scaler.pkl"
features = [
    "price_start_local", "distance_in_meters", "duration_in_seconds",
    "pickup_in_meters", "pickup_in_seconds", "driver_rating",
]

input_dim = len(features)
model_fp32 = PriceNet(input_dim=input_dim)

try:
    model_fp32.load_state_dict(torch.load(MODEL_PATH))
    model_fp32.eval() 
    print(f"Модель успешно загружена из: {MODEL_PATH}")
except FileNotFoundError:
    print(f"Ошибка: Файл модели {MODEL_PATH} не найден. Сначала обучите модель!")
    exit()

model = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear},
    dtype=torch.qint8 
)
print("Модель успешно квантована в INT8.")

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Масштабировщик загружен из: {SCALER_PATH}")
except FileNotFoundError:
    print(f"Ошибка: Файл масштабировщика {SCALER_PATH} не найден!")
    exit()


order = {
    "price_start_local": 200,
    "distance_in_meters": 2000,
    "duration_in_seconds": 900,
    "pickup_in_meters": 800,
    "pickup_in_seconds": 120,
    "driver_rating": 4.4,
}

input_list = [order[f] for f in features]
sample_np = np.array(input_list, dtype=np.float32).reshape(1, -1)

sample_scaled = scaler.transform(sample_np)
sample = torch.tensor(sample_scaled, dtype=torch.float32)



with torch.inference_mode(): 
    price_rel_pred, prob_raw_pred = model(sample)

prob_pred = torch.sigmoid(prob_raw_pred)
price_pred_real = price_rel_pred * order["price_start_local"]

print("\nРекомендованные цены и вероятности согласия:\n")
sorted_indices = torch.argsort(price_pred_real[0]) 

for i in sorted_indices:
    price = price_pred_real[0,i].item()
    prob = prob_pred[0,i].item()
    print(f"Цена: {price:.0f} руб -> Вероятность согласия: {prob*100:.1f}%")

print('%s' % (time.time() - start_time))