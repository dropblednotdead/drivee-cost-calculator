import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==== 1. Загрузка данных ====
df = pd.read_csv("train.csv")

target_price = "price_bid_local"
target_prob = "is_done"

df[target_prob] = df[target_prob].map({'done': 1, 'cancel': 0})

df["price_relative"] = df[target_price] / df["price_start_local"]

# ==== 2. Признаки и целевые ====
features = [
    "price_start_local",
    "distance_in_meters",
    "duration_in_seconds",
    "pickup_in_meters",
    "pickup_in_seconds",
    "driver_rating",
]


# Убираем пропуски
df = df.dropna(subset=features + [target_price, target_prob])

X = df[features].values
y_price = df["price_relative"].values.reshape(-1, 1)
y_prob = df[target_prob].values.reshape(-1, 1)

# ==== 3. Масштабирование ====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==== 4. Разделение данных ====
X_train, X_test, y_price_train, y_price_test, y_prob_train, y_prob_test = train_test_split(
    X_scaled, y_price, y_prob, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_price_train = torch.tensor(y_price_train, dtype=torch.float32)
y_prob_train = torch.tensor(y_prob_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_price_test = torch.tensor(y_price_test, dtype=torch.float32)
y_prob_test = torch.tensor(y_prob_test, dtype=torch.float32)

# ==== 5. Модель ====
class PriceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 3 цены + 3 вероятности, можно расширить
        )

    def forward(self, x):
        out = self.model(x)
        # Делаем 3 цены
        price =  0.8 + 0.7 * torch.sigmoid(out[:, :3])
        # Делаем 3 вероятности через sigmoid
        prob = torch.sigmoid(out[:, 3:])
        price = price * (1 - (prob - 0.5)/2)
        return price, prob

model = PriceNet(input_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss()
bce = nn.BCELoss()

# ==== 6. Обучение ====
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    price_pred, prob_pred = model(X_train)
    # Для старта берём одинаковую цену и вероятность для всех трёх выходов
    y_price_target = y_price_train.repeat(1,3)
    y_prob_target = y_prob_train.repeat(1,3)
    y_prob_target = y_prob_target * 0.5 + 0.5 
    loss = mse(price_pred, y_price_target) + 20.0 * bce(prob_pred, y_prob_target)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# ==== 7. Прогноз для нового заказа ====
order = {
    "price_start_local": 200,
    "distance_in_meters": 2000,
    "duration_in_seconds": 900,
    "pickup_in_meters": 800,
    "pickup_in_seconds": 120,
    "driver_rating": 4.4,
}

sample = torch.tensor(scaler.transform(pd.DataFrame([order])), dtype=torch.float32)
price_rel_pred, prob_pred = model(sample)

# Умножаем на стартовую цену, чтобы получить реальные рубли
price_pred_real = price_rel_pred * order["price_start_local"]

print("\nРекомендованные цены и вероятности согласия:\n")
for i in range(3):
    print(f"Цена: {price_pred_real[0,i].item():.0f} руб -> Вероятность согласия: {prob_pred[0,i].item()*100:.1f}%")
