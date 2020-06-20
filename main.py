import model
import torch.nn as nn
from torch import optim
from utils import DataFromH5File
from torch.utils.data import DataLoader
import torch

"""
    train the customized CNN model, then save it.
    using BCELoss.
    improvement project:
        try to reduce the epochs of training maybe not hurt the performance.
"""
net = model.CNN()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.02)

trainset = DataFromH5File('./data.h5')
train_loader = DataLoader(dataset=trainset, batch_size=1)

i = 0
for epoch in range(20):
    for data in train_loader:
        img, label = data
        img = img.view(1, 1, 150, 150)
        y_hat = net(img)
        loss = criterion(y_hat, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
        if i % 50 == 0:
            print('epoch: {}, loss: {:.4}'.format(i, loss.data.item()))

torch.save(net.state_dict(), "net.pth")
