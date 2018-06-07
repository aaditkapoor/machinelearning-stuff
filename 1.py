
# coding: utf-8

# In[143]:


import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer


# In[144]:


features, labels = load_breast_cancer(return_X_y=True)


# In[145]:


# Building dataset
loader = DataLoader(dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(labels).float()),shuffle=True)


# In[146]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.linear1 = nn.Linear(features.shape[1],50) # Input
        self.linear2 = nn.Linear(50, 25) # Hidden
        self.linear3 = nn.Linear(25, 2) # Output
    def forward(self, x):
        # Activation Functions
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax(dim=1)
        out1 = relu(self.linear1(x))
        out2 = relu(self.linear2(out1))
        output = softmax(self.linear3(out2))
        return output # Final Output between 0 and 1


# In[147]:


classifier = Classifier() # Loading Module


# In[150]:


# Loading parameters
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
epochs = 10


# In[151]:


# Training
for epoch in range(epochs):
    print ("EPOCH #",epoch)
    for id,data in enumerate(loader):
        print (id)
        features, labels = Variable(data[0]), Variable(data[1]).long()
        y_pred = classifier(features)
        loss = loss_fn(y_pred, labels)
        print ("The loss is:", loss.item())
        
        # Zero Gradients
        optimizer.zero_grad()
        loss.backward() # Compute gradients
        optimizer.step() # Update


# In[152]:


# Did not split the data into train and test for the time being
features, labels = load_breast_cancer(return_X_y=True)
features = Variable(torch.from_numpy(features)).float()


# In[153]:


pred = classifier(features)


# In[154]:


pred

