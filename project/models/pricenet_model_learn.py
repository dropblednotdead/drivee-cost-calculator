import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/train.csv")

target_price = "price_bid_local"
target_prob = "is_done"

df[target_prob] = df[target_prob].map({'done': 1, 'cancel': 0})

df["price_relative"] = df[target_price] / df["price_start_local"]

features = [
    "price_start_local",
    "distance_in_meters",
    "duration_in_seconds",
    "pickup_in_meters",
    "pickup_in_seconds",
    "driver_rating",
]

df = df.dropna(subset=features + [target_price, target_prob])

X = df[features].values
y_price = df["price_relative"].values.reshape(-1, 1)
y_prob = df[target_prob].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import joblib 
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

model = PriceNet(input_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss()

num_accepted = y_prob_train.sum().item()
num_total = len(y_prob_train)
num_cancelled = num_total - num_accepted

POSITIVE_WEIGHT = torch.tensor(5.0, dtype=torch.float32)
bce = nn.BCEWithLogitsLoss(pos_weight=POSITIVE_WEIGHT)

PROB_LOSS_WEIGHT = 40.0 

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    price_pred, prob_raw_pred = model(X_train)
    
    y_price_target = y_price_train.repeat(1,3) 
    
    
    y_prob_target = y_prob_train.repeat(1,3) 
    
   
    loss_price = mse(price_pred, y_price_target)
    
    loss_prob = bce(prob_raw_pred, y_prob_target    )
    
    loss = loss_price + PROB_LOSS_WEIGHT * loss_prob 
    
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {loss.item():.4f} (Price: {loss_price.item():.4f}, Prob BCE: {loss_prob.item():.4f})")

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    price_pred, prob_pred = model(X_train)
    
    y_price_target = y_price_train.repeat(1,3) 
    
    y_prob_target = y_prob_train.repeat(1,3) 
    
    loss_price = mse(price_pred, y_price_target)
    loss_prob = bce(prob_pred, y_prob_target)
    
    loss = loss_price + 5.0 * loss_prob
    
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {loss.item():.4f} (Price: {loss_price.item():.4f}, Prob: {loss_prob.item():.4f})")

MODEL_PATH = "pricenet_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nМодель успешно сохранена в: {MODEL_PATH}")

order = {
    "price_start_local": 100,
    "distance_in_meters": 1000,
    "duration_in_seconds": 900,
    "pickup_in_meters": 800,
    "pickup_in_seconds": 120,
    "driver_rating": 4.7,
}

sample_df = pd.DataFrame([order], columns=features)
sample_scaled = scaler.transform(sample_df)
sample = torch.tensor(sample_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    price_rel_pred, prob_pred = model(sample)


prob_pred = torch.sigmoid(prob_raw_pred)

price_pred_real = price_rel_pred * order["price_start_local"]

print("\nРекомендованные цены и вероятности согласия (ИСПРАВЛЕННАЯ МОДЕЛЬ):\n")

sorted_indices = torch.argsort(price_pred_real[0]) 

for i in sorted_indices:
    price = price_pred_real[0,i].item()
    prob = prob_pred[0,i].item()
    print(f"Цена: {price:.0f} руб -> Вероятность согласия: {prob*100:.1f}%")