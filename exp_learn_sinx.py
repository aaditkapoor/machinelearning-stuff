
# coding: utf-8

# In[9]:


# I will make the neural network the f(x) = sin(x)
# author: aadit
import torch
import random
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# f_x -> function to learn
def f_x(x):
    return np.sin(x)

# Build dataset
def build_dataset():
    data = []
    for i in range(1,300):
        data.append((i, f_x(i), 1)) # 1 stands for "correct value"
    for j in range(300, 600):
        data.append((j, np.cos(j), 0)) # 0 stands for "incorrect value"
    random.shuffle(data)
    df = pd.DataFrame(data=data, columns=["x", "f_x", "is_f_x"])
    return df


# In[5]:


df = build_dataset()


# In[6]:


df.head()


# In[7]:


# extracting labels and features
labels = df.is_f_x.values
features = df.drop(columns=['is_f_x']).values


# In[8]:


print ("shape of features: ", features.shape)
print ("shape of labels: ", labels.shape)


# In[83]:


# Building model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 75)
        self.fc4 = nn.Linear(75, 1)
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.sigmoid(self.fc4(out))
        return out

model = Model()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()
epochs = 1000


# In[84]:


def plot_loss(losses, range=range(1, epochs)):
    assert len(losses) == len(range), "Size is not same!"
    plt.plot(range, losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

def cal_acc(y,y_pred):
    correct = y.eq(y_pred).sum()
    correct = correct.data.numpy()[0]
    return correct/total * 100


# In[85]:


# Splitting
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, shuffle=True, random_state=34)


# In[86]:


x_train = Variable(torch.from_numpy(features_train)).float()
y_train = Variable(torch.from_numpy(labels_train)).float()

x_test = Variable(torch.from_numpy(features_test)).float()
y_test = Variable(torch.from_numpy(labels_test)).float()


# In[87]:


def train():
    losses = []
    model.train()
    for epoch in range(1, epochs):
        optimizer.zero_grad()
        print ("epoch #", epoch)
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train.view(-1,1))
        print ("loss: ", loss.item())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


# In[88]:


losses = train()


# In[89]:


plot_loss(losses)


# In[90]:


y_pred = model(x_test)


# In[91]:


y_pred = torch.round(y_pred)


# In[92]:


from sklearn.metrics import accuracy_score


# In[93]:


accuracy_score(y_test.data.numpy(), y_pred.data.numpy())


# In[99]:


y_p_test = model(x_train)


# In[98]:


y_p_test = torch.round(y_p_test)


# In[97]:


accuracy_score(y_train.data.numpy(), y_p_test.data.numpy())


# In[115]:


x = torch.Tensor([4533])
y = torch.sin(x)
i = torch.Tensor([x,1212])


# In[116]:


i


# In[117]:


torch.round(model(i))

