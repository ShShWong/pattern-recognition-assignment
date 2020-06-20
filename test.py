import torch
from model import CNN
import utils
from torch.utils.data import DataLoader

"""
    test the perfomance of the model on test set
"""

model = CNN()

model.load_state_dict(torch.load("net.pth"))

count = 0
testset = utils.TestFromH5File('./data.h5')
test_loader = DataLoader(dataset=testset, batch_size=1)
for idx, (X_test, y_test) in enumerate(test_loader):
    X_test = X_test.view(1, 1, 150, 150)
    predict = model(X_test)
    if predict > 0.5:
        predict = 1.0
    if predict < 0.5:
        predict = 0.0
    if predict == float(y_test):
        count += 1

accuracy = count / 200
print(accuracy)
