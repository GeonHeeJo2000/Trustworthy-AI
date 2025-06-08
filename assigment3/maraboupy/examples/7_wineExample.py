"""
End-to-end Marabou Example using PyTorch and Wine Quality Data

Author: Geonhee Jo
Date: June 8, 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from maraboupy import Marabou, MarabouNetworkONNX

# -----------------------------
# Step 1: Load and preprocess dataset
# -----------------------------
print("[INFO] Loading winequality-red.csv...")
df = pd.read_csv("winequality-red.csv", sep=";")
X = df.drop("quality", axis=1).values
y = (df["quality"] >= 6).astype(np.float32).values  # Binary classification

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# -----------------------------
# Step 2: Define PyTorch model
# -----------------------------
class WineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

model = WineNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Step 3: Train the model
# -----------------------------
print("[INFO] Training PyTorch model...")
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    output = model(X_train)
    
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
print("[INFO] Training complete.")

# -----------------------------
# Step 4: Export to ONNX
# -----------------------------
print("[INFO] Exporting model to ONNX...")
dummy_input = torch.randn(1, 11)
onnx_path = "wine_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"], opset_version=11)
print(f"[INFO] Model saved to {onnx_path}")

# -----------------------------
# Step 5: Load into Marabou and verify
# -----------------------------
# Set the Marabou option to restrict printing
options = Marabou.createOptions(verbosity = 0)

print("[INFO] Verifying with Marabou (ONNX)...")
#net = MarabouNetworkONNX(onnx_path)
network = Marabou.read_onnx(onnx_path)
# Constraint: alcohol (x_10) ∈ [0.0, 0.5] → output ≥ 0.8

# %%
# Get the input and output variable numbers; [0] since first dimension is batch size
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0][0]

# %%
# Set input bounds
network.setLowerBound(inputVars[0],-10.0)
network.setUpperBound(inputVars[0], 10.0)
network.setLowerBound(inputVars[1],-10.0)
network.setUpperBound(inputVars[1], 10.0)

# %%
# Set output bounds
network.setLowerBound(outputVars[0], 194.0)
network.setUpperBound(outputVars[0], 210.0)

#network.setUpperBound(outputVars[0], 0.5)

# Solve
exit_code, vals, stats = network.solve(options = options)
