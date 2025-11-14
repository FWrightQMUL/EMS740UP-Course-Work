import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("train_reduced_gB4.csv")
Y_train = pd.read_csv("train_target_gB4.csv")

X_val = pd.read_csv("validation_reduced_gB4.csv")
Y_val = pd.read_csv("validation_target_gB4.csv")

X_test = pd.read_csv("test_reduced_gB4.csv")
Y_test = pd.read_csv("test_target_gB4.csv")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_val_scaled = scaler.transform(X_val.values)
X_test_scaled = scaler.transform(X_test.values)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

Y_train_t = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
Y_val_t = torch.tensor(Y_val.values, dtype=torch.float32).view(-1, 1)
Y_test_t = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_t, Y_train_t)
val_ds = TensorDataset(X_val_t, Y_val_t)
test_ds = TensorDataset(X_test_t, Y_test_t)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)

n_features = X_train_t.shape[1]

model = nn.Sequential(
    nn.Linear(n_features, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

best_val = float('inf')
patience = 20
trigger = 0

EPOCHS = 200

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for Xb, Yb in train_loader:
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, Yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, Yb in val_loader:
            preds = model(Xb)
            val_loss += criterion(preds, Yb).item()

    scheduler.step(val_loss)
    print(f"{epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        trigger = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        trigger += 1

    if trigger >= patience:
        print("Early stopping.")
        break

model.load_state_dict(torch.load("best_model.pth"))

model.eval()
test_loss = 0
with torch.no_grad():
    for Xb, Yb in test_loader:
        preds = model(Xb)
        test_loss += criterion(preds, Yb).item()

print("Final Test MSE:", test_loss)

preds = model(X_test_t).detach().numpy()
pd.DataFrame(preds, columns=["pred_band_gap"]).to_csv("NN_pred_gB4.csv", index=False)

preds = pd.read_csv("NN_pred_gB4.csv")
true = pd.read_csv("test_target_gB4.csv")
pred_vals = preds.values.flatten()
true_vals = true.values.flatten()

plt.figure(figsize=(6, 6))
plt.scatter(true_vals, pred_vals, alpha=0.6)
min_val = min(true_vals.min(), pred_vals.min())
max_val = max(true_vals.max(), pred_vals.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
plt.xlabel("True band_gap")
plt.ylabel("Predicted band_gap")
plt.title("Predicted vs True")
plt.grid(True)
plt.show()

residuals = pred_vals - true_vals
plt.figure(figsize=(6, 4))
plt.scatter(true_vals, residuals, alpha=0.6)
plt.axhline(0, linestyle='--', color='black')
plt.xlabel("True band_gap")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.hist(true_vals, bins=30, alpha=0.6, label="True")
plt.hist(pred_vals, bins=30, alpha=0.6, label="Predicted")
plt.legend()
plt.xlabel("band_gap")
plt.ylabel("Count")
plt.title("Distribution Comparison")
plt.grid(True)
plt.show()
