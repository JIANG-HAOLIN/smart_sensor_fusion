import math

import torch
import torch.nn as nn
import torch.optim as optim


print(f"if Cuda available:{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Cuda info:\n{torch.cuda.get_device_properties('cuda')}")
    print(f"Cuda version:{torch.version.cuda}")
else:
    print(f'no Cuda detected, using CPU instead !!')

w = torch.nn.Parameter(torch.randn([1, ], device='cuda'), requires_grad=True)
y = torch.tensor(1, device='cuda')

loss = y - w
grad = loss.backward()
print("gradient computed successfully!!", w.grad)


# Generate synthetic data
torch.manual_seed(42)  # for reproducibility
X = torch.linspace(0, 99, 100, device='cuda').reshape(-1, 1)
y_true = torch.sin(X)
y_noisy = y_true + torch.randn_like(y_true, device='cuda')

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 10), nn.ReLU(), nn.Linear(10, 1))

    def forward(self, x):
        return self.MLP(x)

# Instantiate the model, loss function, and optimizer
model = SimpleModel().to('cuda')
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X)

    # Compute the loss
    loss = criterion(y_pred, y_noisy)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 1 == 0:
        print(f'GPU and pytorch are working correctly! Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

