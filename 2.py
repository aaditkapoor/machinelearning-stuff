
# coding: utf-8

# In[201]:


import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer


# In[202]:


features, labels = load_breast_cancer(return_X_y=True)


# In[203]:


# Building dataset
loader = DataLoader(dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(labels).float()),shuffle=True)


# In[204]:


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


# In[205]:


classifier = Classifier() # Loading Module


# In[206]:


# Loading parameters
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
epochs = 50


# In[207]:


features = Variable(torch.from_numpy(features)).float()
labels = Variable(torch.from_numpy(labels)).long()
# Training
for epoch in range(epochs):
    print ("EPOCH #",epoch)
    y_pred = classifier(features)
    loss = loss_fn(y_pred,labels)
    print ("The loss is:", loss.item())
        
        # Zero Gradients
    optimizer.zero_grad()
    loss.backward() # Compute gradients
    optimizer.step() # Update


# In[208]:


features


# In[209]:


pred = classifier(features)


# In[214]:


pred = np.argmax(pred.detach().numpy(), axis=1)


# In[215]:


accuracy_score(labels, pred)

