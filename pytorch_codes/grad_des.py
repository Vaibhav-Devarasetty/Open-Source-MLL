import torch
from torch import nn
from torch.optim import SGD

#making data ready for y = 2*x function
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
learning_rate = 0.1
w = torch.tensor(0, dtype=torch.float32, requires_grad=True)
max_epochs = 10

def pred(x):
    return x * w

loss = nn.MSELoss()
optimizer = SGD([w], lr=learning_rate)

for epoch in range(max_epochs):
    #forward and calculate loss
    y_pred = pred(x)
    l =loss(y_pred, y)

    #backprop for grad cal
    optimizer.zero_grad()
    l.backward()

    #gradient descent for optimization(decreasing loss)
    optimizer.step()
    with torch.no_grad():
        print(f"Epoch {epoch+1} : w value : {w}")

w = w.detach()
print(pred(5).item())