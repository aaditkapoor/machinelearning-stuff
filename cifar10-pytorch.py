# author: aadit

import torch
from torch.autograd import Variable
from torch import nn
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

torch.random.manual_seed(1)

# Loading data
transforms_list_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
transforms_list_test = transforms.Compose([transforms.ToTensor()])

train_set = CIFAR10("./cifar10-data-imple-train", train=True, transform=transforms_list_train, download=True)
test_set =  CIFAR10("./cifar10-data-imple-test", train=False, transform=transforms_list_test, download=True)

# Loading loaders
train_loader = DataLoader(dataset=train_set, batch_size=200, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)


# model def
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0),-1) # Flatten the tensor


net = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),nn.Dropout(0.2), nn.Conv2d(16, 64, kernel_size=5), nn.ReLU(),nn.Dropout(0.2), nn.MaxPool2d(2), Flatten(), nn.Linear(1600, 100), nn.ReLU(), nn.Linear(100, 10))
print ("Printing model summary")
print (net)

print

optimizer = torch.optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()
epochs=300


def train(epochs):
    print ("training......")
    net.train()
    for epoch in range(1, epochs+1):
        print ("epoch #", epoch)
        current_loss = 0.0
        current_acc = 0.0
        for feature, label in train_loader:
            optimizer.zero_grad()
            x = Variable(feature, requires_grad=False).float()
            y = Variable(label, requires_grad=False).long()
            
            y_pred = net(x)
            loss = criterion(y_pred, y)
            #print ("loss: ", loss.item())
            current_acc = accuracy_score(y.data.numpy(), y_pred.max(1)[1].data.numpy())
            current_loss+=loss
            #print ("current acc: ", current_acc.item())
            loss.backward()
            optimizer.step()
        #print ("loss: ", current_loss.item())

train(100)

            
