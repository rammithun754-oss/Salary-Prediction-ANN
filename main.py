import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

X = np.array([
    [0], [1], [2], [3], [4],
    [5], [6], [7], [8], [9], [10]
], dtype=float)

y = np.array([
    [20], [25], [30], [35], [40],
    [45], [50], [55], [60], [65], [70]
], dtype=float)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

class SalaryANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden = nn.Linear(1, 4)

        self.output = nn.Linear(4, 1)

    def forward(self, x):

        x = torch.relu(self.hidden(x))

        x = self.output(x)

        return x

model = SalaryANN()

criterion = nn.MSELoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)

epochs = 1000

for epoch in range(epochs):

    predictions = model(X)

    loss = criterion(predictions, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 200 == 0:

        print(
            f"Epoch {epoch} | "
            f"Loss: {loss.item():.4f}"
        )

test_input = torch.tensor(
    [[6.5]],
    dtype=torch.float32
)

with torch.no_grad():

    predicted_salary = model(test_input)

print(
    "\nPredicted Salary (in thousands):",
    predicted_salary.item()
)
