import torch
import matplotlib.pyplot as plt
import torchvision

#device_config
if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#hyperparameters
learning_rate = 0.01
max_epochs = 2
batch_size = 100
input_size = 28*28
hidden_size = 128
output_size = 10

#making dataset ready

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


#model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x