import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_MODEL_PATH = "base_model.pkl"
BASE_MODEL_FEATURES = ["distance_in_meters", "duration_in_seconds"]

df = pd.read_csv("./train.csv")

target_price = "price_bid_local"
target_prob = "is_done"

df[target_prob] = df[target_prob].map({'done': 1, 'cancel': 0})

df['order_dt'] = pd.to_datetime(df['order_timestamp'])
df['tender_dt'] = pd.to_datetime(df['tender_timestamp'])
time_diff = df['tender_dt'] - df['order_dt']
df['waiting_time_sec'] = time_diff.dt.total_seconds().clip(lower=0) 
df['hour_of_day'] = df['order_dt'].dt.hour
df['day_of_week'] = df['order_dt'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

df.drop(columns=['order_timestamp', 'tender_timestamp', 'order_dt', 'tender_dt', 'hour_of_day'], inplace=True, errors='ignore')

try:
    base_model = joblib.load(BASE_MODEL_PATH)
    print(f"Базовая модель (Linear Regression) загружена для инженерии признаков.")
except FileNotFoundError:
    print(f"Ошибка: Файл базовой модели {BASE_MODEL_PATH} не найден. Сначала обучите ее!")
    exit()

df_for_base = df.dropna(subset=BASE_MODEL_FEATURES + ["price_start_local"])

X_base = df_for_base[BASE_MODEL_FEATURES]
df_for_base.loc[:, 'predicted_base_price'] = base_model.predict(X_base)

df_for_base.loc[:, 'price_anomaly'] = (df_for_base['predicted_base_price'] - df_for_base["price_start_local"]) / df_for_base['predicted_base_price']

df = pd.merge(df, df_for_base[['price_anomaly']], left_index=True, right_index=True, how='left')

df["price_relative"] = df[target_price] / df["price_start_local"]

np.random.seed(42) 
MIN_RATING = 4.5
MAX_RATING = 5.0
MAX_PRICE_ADJUSTMENT = 0.02

df['user_rating'] = np.random.uniform(MIN_RATING, MAX_RATING, size=len(df))
rating_norm = (df['user_rating'] - MIN_RATING) / (MAX_RATING - MIN_RATING)
price_multiplier = 1.0 - (rating_norm * MAX_PRICE_ADJUSTMENT)
df["price_relative"] = df["price_relative"] * price_multiplier

features = [
    "price_start_local",
    "distance_in_meters",
    "duration_in_seconds",
    "pickup_in_meters",
    "pickup_in_seconds",
    "driver_rating",
    "waiting_time_sec",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "price_anomaly",
    "user_rating", 
]

df = df.dropna(subset=features + [target_price, target_prob])

X = df[features].values
y_price = df["price_relative"].values.reshape(-1, 1)
y_prob = df[target_prob].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl") 
print("Масштабировщик сохранен.")

X_train, X_test, y_price_train, y_price_test, y_prob_train, y_prob_test = train_test_split(
    X_scaled, y_price, y_prob, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_price_train = torch.tensor(y_price_train, dtype=torch.float32)
y_prob_train = torch.tensor(y_prob_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_price_test = torch.tensor(y_price_test, dtype=torch.float32)
y_prob_test = torch.tensor(y_prob_test, dtype=torch.float32)

class ResidualBlock(nn.Module):
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
        
        # SHARED BODY (Соответствует обученному: ResNet)
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
            nn.Dropout(0.2), # Слой Dropout - важен, иначе счет слоев будет другим!
            nn.Linear(hidden_dim // 2, num_recommendations) 
        )

        # ГОЛОВЫ (Соответствует обученному: Простой Linear, т.к. сложные были перезаписаны)
        # Мы оставляем только финальную, перезаписанную версию!

        # Адаптивные веса потерь (должны быть, чтобы ключи совпали)
        self.log_price_variance = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_prob_variance = nn.Parameter(torch.zeros(1), requires_grad=True)
        
    def forward(self, x):
        shared_out = self.shared_body(x)
        
        # В forward используем простые linear-слои
        price_raw = self.price_head(shared_out)
        price = 0.8 + 0.7 * torch.sigmoid(price_raw) 
        prob_raw = self.prob_head(shared_out)
        return price, prob_raw

model = PriceNet(input_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss()

POSITIVE_WEIGHT = torch.tensor(5.0, dtype=torch.float32)
bce = nn.BCEWithLogitsLoss(pos_weight=POSITIVE_WEIGHT)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    price_pred, prob_raw_pred = model(X_train)
    
    y_price_target = y_price_train.repeat(1,3) 
    y_prob_target = y_prob_train.repeat(1,3) 

    loss_price = mse(price_pred, y_price_target)
    loss_prob = bce(prob_raw_pred, y_prob_target)

    s_price = model.log_price_variance
    s_prob = model.log_prob_variance

    term_price = torch.exp(-s_price) * loss_price + 0.5 * s_price
    term_prob = torch.exp(-s_prob) * loss_prob + 0.5 * s_prob

    loss = term_price + term_prob

    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        s_price = model.log_price_variance.item()
        s_prob = model.log_prob_variance.item()
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {loss.item():.4f} | Vars (Price/Prob): {s_price:.2f} / {s_prob:.2f}")


epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    price_pred, prob_raw_pred = model(X_train)

    y_price_target = y_price_train.repeat(1,3) 
    y_prob_target = y_prob_train.repeat(1,3) 

    loss_price = mse(price_pred, y_price_target)
    loss_prob = bce(prob_raw_pred, y_prob_target) 

    s_price = model.log_price_variance
    s_prob = model.log_prob_variance

    term_price = torch.exp(-s_price) * loss_price + 0.5 * s_price
    term_prob = torch.exp(-s_prob) * loss_prob + 0.5 * s_prob

    loss = term_price + term_prob

    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        s_price = model.log_price_variance.item()
        s_prob = model.log_prob_variance.item()
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {loss.item():.4f} | Vars (Price/Prob): {s_price:.2f} / {s_prob:.2f}")


MODEL_PATH = "pricenet_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nМодель успешно сохранена в: {MODEL_PATH}")

WAITING_TIME_MODEL_PATH = "waiting_time_model.pkl"
WAITING_TIME_MODEL_FEATURES = [
    "distance_in_meters",
    "duration_in_seconds",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "pickup_in_meters",
]

try:
    waiting_time_model = joblib.load(WAITING_TIME_MODEL_PATH)
    print(f"Модель времени ожидания загружена.")
except FileNotFoundError:
    print(f"Ошибка: Файл модели времени ожидания {WAITING_TIME_MODEL_PATH} не найден! Сначала обучите ее!")
    exit()

order = {
    "order_timestamp": "2025-01-13 08:15:00",
    "price_start_local": 100,
    "distance_in_meters": 1000,
    "duration_in_seconds": 900,
    "pickup_in_meters": 800,
    "pickup_in_seconds": 120,
    "driver_rating": 4.7,
    "user_rating": 4.95,
}

order_dt = pd.to_datetime(order['order_timestamp'])
order['hour_of_day'] = order_dt.hour
order['day_of_week'] = order_dt.dayofweek
order['hour_sin'] = np.sin(2 * np.pi * order['hour_of_day'] / 24)
order['hour_cos'] = np.cos(2 * np.pi * order['hour_of_day'] / 24)

time_input_list = [order[f] for f in WAITING_TIME_MODEL_FEATURES]
time_sample_np = np.array(time_input_list, dtype=np.float32).reshape(1, -1)
predicted_waiting_time = waiting_time_model.predict(time_sample_np)[0]

base_price_input_list = [order[f] for f in BASE_MODEL_FEATURES]
base_price_sample_np = np.array(base_price_input_list, dtype=np.float32).reshape(1, -1)
predicted_base_price = base_model.predict(base_price_sample_np)[0]

price_anomaly = (predicted_base_price - order["price_start_local"]) / predicted_base_price
order["price_anomaly"] = price_anomaly 
print(f"Аномалия цены (price_anomaly): {price_anomaly:.3f}")

order["waiting_time_sec"] = predicted_waiting_time
print(f"Прогноз времени ожидания: {predicted_waiting_time:.1f} сек.")

sample_np = np.array([order[f] for f in features], dtype=np.float32).reshape(1, -1)

sample_scaled = scaler.transform(sample_np)
sample = torch.tensor(sample_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    price_rel_pred, prob_raw_pred = model(sample)


prob_pred = torch.sigmoid(prob_raw_pred)

price_pred_real = price_rel_pred * order["price_start_local"]

print("\nРекомендованные цены и вероятности согласия (ФИНАЛЬНЫЙ ПРОГНОЗ):\n")

sorted_indices = torch.argsort(price_pred_real[0]) 

for i in sorted_indices:
    price = price_pred_real[0,i].item()
    prob = prob_pred[0,i].item()
    print(f"Цена: {price:.0f} руб -> Вероятность согласия: {prob*100:.1f}%")