import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

batch_size = 64
device = torch.device("cpu")

x = torch.randn(batch_size, 1000, device=device, dtype=torch.float)
y = torch.randn(batch_size, 10, device=device, dtype=torch.float)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(1000,100)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(100,10)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x

writer = SummaryWriter('./events')
lr = 1e-4
model = Net()
mse_loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9)

for i in range(5000):
  output = model(x)
  loss = mse_loss(output, y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  writer.add_scalar('Loss', loss, i)
  if i % 1000 == 0:
    print("epoch ", i)
    print("loss ", loss.item())
writer.close()