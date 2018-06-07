
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import torch.nn.functional as F


# In[2]:


data = pd.read_csv("/Users/aaditkapoor/Documents/sms.tsv", delimiter='\t', header=None, names=["outcome", 'message'])


# In[3]:


data.head()


# In[4]:


data.outcome = data.outcome.map({'ham':0, 'spam':1})


# In[5]:


data.head()


# In[13]:


features = data.message.values
labels = data.outcome.values
num_words = 1000


# In[14]:


features.shape


# In[15]:


labels.shape


# In[16]:


t = Tokenizer(num_words=1000)
t.fit_on_texts(features)


# In[17]:


features = t.texts_to_matrix(features, mode='tfidf')


# In[19]:


features.shape


# In[21]:


# Building model
class Model(nn.Module):
    def __init__(self, input, hidden, output):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input, hidden)
        self.l2 = nn.Linear(hidden , hidden)
        self.l3 = nn.Linear(hidden, 2)
    
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return out        


# In[22]:


input = 1000
hidden=100
output = 2


# In[23]:


model = Model(input, hidden, output)


# In[24]:


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, shuffle=True, random_state=34)


# In[26]:


# params
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


# In[43]:


def train(epochs):
    x_train = Variable(torch.from_numpy(features_train)).float()
    y_train = Variable(torch.from_numpy(labels_train)).long()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        print ("epoch #",epoch)
        print ("loss: ", loss.item())
        pred = torch.max(y_pred, 1)[1].eq(y_train).sum()
        print ("acc:(%) ", 100*pred/len(x_train))
        loss.backward()
        optimizer.step()


# In[44]:


def test(epochs):
    model.eval()
    x_test = Variable(torch.from_numpy(features_test)).float()
    y_test = Variable(torch.from_numpy(labels_test)).long()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_test)
        loss = criterion(y_pred, y_test)
        print ("epoch #",epoch)
        print ("loss: ", loss.item())
        pred = torch.max(y_pred, 1)[1].eq(y_test).sum()
        print ("acc (%): ", 100*pred/len(x_test))
        loss.backward()
        optimizer.step()


# In[45]:


train(100)


# In[46]:


test(100)


# In[49]:


pred = model(torch.from_numpy(features_test).float())


# In[50]:


pred


# In[51]:


pred = torch.max(pred,1)[1]


# In[53]:


len(pred)


# In[54]:


len(features_test)


# In[55]:


pred = pred.data.numpy()


# In[56]:


pred


# In[57]:


labels_test


# In[58]:


accuracy_score(labels_test, pred)


# In[59]:


p_train = model(torch.from_numpy(features_train).float())


# In[60]:


p_train = torch.max(p_train,1)[1]


# In[61]:


len(p_train)


# In[62]:


p_train = p_train.data.numpy()


# In[63]:


p_train


# In[67]:


accuracy_score(labels_train, p_train)


# In[68]:


from sklearn.metrics import confusion_matrix


# In[69]:


cm = confusion_matrix(labels_test, pred)


# In[70]:


print (cm)

