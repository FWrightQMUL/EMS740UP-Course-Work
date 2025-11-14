import pandas as pd
import torch
from torch import nn
from torch.nn.functional import dropout
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import matplotlib.pyplot as plt


X_train = pd.read_csv("train_reduced_gB4.csv")
Y_train = pd.read_csv("train_target_gB4.csv")

X_val = pd.read_csv("validation_reduced_gB4.csv")
Y_val = pd.read_csv("validation_target_gB4.csv")

X_test = pd.read_csv("test_reduced_gB4.csv")
Y_test = pd.read_csv("test_target_gB4.csv")


X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)

X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val.values, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)


train_ds = TensorDataset(X_train_t, Y_train_t)
val_ds   = TensorDataset(X_val_t, Y_val_t)
test_ds  = TensorDataset(X_test_t, Y_test_t)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 256),
    nn.ReLU(),
    nn.Dropout(0.1),

    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50

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

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, Yb in val_loader:
            preds = model(Xb)
            val_loss += criterion(preds, Yb).item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

model.eval()
test_loss = 0

with torch.no_grad():
    for Xb, Yb in test_loader:
        preds = model(Xb)
        test_loss += criterion(preds, Yb).item()

print("Final Test MSE:", test_loss)


model.eval()
preds = model(X_test_t).detach().numpy()

pd.DataFrame(preds, columns=["pred_band_gap"]).to_csv("NN_pred_gB4.csv", index=False)


# Load predictions and true values
preds = pd.read_csv("NN_pred_gB4.csv")
true  = pd.read_csv("test_target_gB4.csv")

plt.figure(figsize=(6, 6))
plt.scatter(true, preds, alpha=0.6)
plt.plot([true.min()[0], true.max()[0]],
         [true.min()[0], true.max()[0]],
         linestyle='--')  # 1:1 line

plt.xlabel("True band_gap")
plt.ylabel("Predicted band_gap")
plt.title("Neural Network Predictions vs True Values")
plt.grid(True)
plt.show()

residuals = preds.values.flatten() - true.values.flatten()

plt.figure(figsize=(6, 4))
plt.scatter(true, residuals, alpha=0.6)
plt.axhline(0, linestyle='--', color='black')

plt.xlabel("True band_gap")
plt.ylabel("Prediction Error (Pred - True)")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))

plt.hist(true, bins=30, alpha=0.6, label="True")
plt.hist(preds, bins=30, alpha=0.6, label="Predicted")

plt.legend()
plt.xlabel("band_gap")
plt.ylabel("Count")
plt.title("True vs Predicted Distribution")
plt.grid(True)
plt.show()