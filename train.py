"""
Training script for Physics-Informed Neural Network (PINN).

Solves the differential equation dy/dt + y = 0 with y(0) = 1
and visualizes the training loss and final solution.

Usage:
    python train.py [N_EPOCHS] [LR]
    
Example:
    python train.py 10000 0.001
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pinn import PINN

T_MAX = 1.0
N_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
N_SAMPLES = N_EPOCHS // 10

t_physics = torch.linspace(0, T_MAX, N_SAMPLES).unsqueeze(1)
t_physics.requires_grad = True 

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
losses = []

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    loss = model.loss(t_physics)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses, linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss over Epochs')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

t_test = torch.linspace(0, T_MAX, 200).unsqueeze(1)
with torch.no_grad():
    y_pred = model(t_test).numpy()
t_np = t_test.numpy().flatten()
y_true = np.exp(-t_np)

axes[1].plot(t_np, y_true, 'b-', label='Analytical Solution', linewidth=2)
axes[1].plot(t_np, y_pred, 'r--', label='PINN Prediction', linewidth=2)
axes[1].set_xlabel('t')
axes[1].set_ylabel('y(t)')
axes[1].set_title('PINN Solution vs Analytical Solution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

fig.text(0.5, 0.01, f'Epochs: {N_EPOCHS}  |  Learning Rate: {LR}', 
         ha='center', va='bottom', fontsize=10, style='italic', color='#555')

timestamp = datetime.now().strftime('%m%d%y_%H%M%S')
filename = f'pinn_results_{timestamp}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"\nPlots saved to '{filename}'")
plt.show()
