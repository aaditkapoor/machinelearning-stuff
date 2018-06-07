import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer

use_bce = False

features, labels = load_breast_cancer(return_X_y=True)
features = Variable(torch.from_numpy(features)).float()
labels = Variable(torch.from_numpy(labels))
labels = labels.float() if use_bce else labels.long()

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.net = nn.Sequential(nn.Linear(features.shape[1],50), nn.ReLU(), nn.Linear(50, 25), nn.ReLU(), nn.Linear(25, 1 if use_bce else 2))
    def forward(self, x):
        return self.net(x).squeeze()

classifier = Classifier()
optimizer = optim.Adam(classifier.parameters(), lr=0.0001, weight_decay=0.1)
loss_fn = nn.BCEWithLogitsLoss() if use_bce else nn.CrossEntropyLoss()
epochs = 1000

def get_acc():
    if use_bce:
        return ((classifier(features) > 0.5) == labels.byte()).float().mean().item()*100.0
    return (torch.argmax(classifier(features), dim=1) == labels).float().mean().item()*100.0

for epoch in range(epochs):
    y_pred = classifier(features)
    loss = loss_fn(y_pred,labels)
    print('epoch: {}, loss: {}, acc: {}'.format(epoch, loss.item(), get_acc()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
