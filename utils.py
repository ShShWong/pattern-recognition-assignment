from torch.utils.data import DataLoader, Dataset
import h5py

"""
generate dataset from h5 file
"""


class DataFromH5File(Dataset):
    def __init__(self, filepath):
        h5File = h5py.File(filepath, 'r')
        self.X_train = h5File['X_train']
        self.y_train = h5File['y_train']
        self.X_test = h5File['X_test']
        self.y_test = h5File['y_test']

    def __getitem__(self, item):
        label = self.y_train[item]
        data = self.X_train[item]
        return data, label

    def __len__(self):
        assert self.X_train.shape[0] == self.y_train.shape[0], "Wrong data length"
        return self.X_train.shape[0]


class TestFromH5File(Dataset):
    def __init__(self, filepath):
        h5File = h5py.File(filepath, 'r')
        self.X_test = h5File['X_test']
        self.y_test = h5File['y_test']

    def __getitem__(self, item):
        label = self.y_test[item]
        data = self.X_test[item]
        return data, label

    def __len__(self):
        assert self.X_test.shape[0] == self.y_test.shape[0], "Wrong data length"
        return self.X_test.shape[0]

# usage:
# trainset = DataFromH5File('./data.h5')
# print(trainset[2])
# train_loader = DataLoader(dataset=trainset, batch_size=8)
# for idx, (X_train, y_train) in enumerate(train_loader):
#     print(X_train.shape)
#     print(y_train)
#     break


