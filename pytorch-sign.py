import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch import  nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


features = np.load("/Users/aaditkapoor/Downloads/Sign-language-digits-dataset 2/X.npy")
labels = np.load("/Users/aaditkapoor/Downloads/Sign-language-digits-dataset 2/Y.npy")

print ("SHAPES")
print (features.shape)
print (labels.shape)


features = features.reshape(2062, -1)
pca = PCA(n_components=1000)
features = pca.fit_transform(features)


print ("Reshaped..")
print (features.shape)
print (labels.shape)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(1000, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 100)
        self.l4 = nn.Linear(100, 10)
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = (self.l4(out))
        return out


model = Model()
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=34, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
def train(epochs):
	model.train()
	x = Variable(torch.from_numpy(features_train), requires_grad=False).float()

	y = Variable(torch.from_numpy(labels_train), requires_grad=False).long()
	
	for epoch in range(epochs):
		optimizer.zero_grad()
		pred = model(x)
		pred = torch.max(pred, 1)[1]
		loss = criterion(output.float(), pred)

		print ("epoch #", epoch)
		print ("loss: ", loss.item())
		loss.backward()
		optimizer.step()

def test(epochs):
	model.eval()
	x = Variable(torch.from_numpy(features_test)).float()
	y = Variable(torch.from_numpy(labels_test)).long()
	with torch.no_grad():
		for e in epochs:
			loss = criterion(model(x), y)
			print ("loss: ", loss)

