import torch
import h5py
import numpy as np
DATA_PATH = 'monocolor/types_100/'
IMG_DIR = 'card_images/'
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, ids_file, labels_file, transform=None):
        self.transform = transform
        self.labels = np.load(labels_file)
        self.ids = np.load(ids_file)
        self.labels = np.load(labels_file)
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        single_id = self.ids[idx]
        data = mpimg.imread(IMG_DIR + single_id + '.png')
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)
        return data, label


data_train = ImageDataset(DATA_PATH + 'ids.npy', DATA_PATH + 'colors.npy')
data, label = data_train[0]
imgplot = plt.imshow(data)
print(label)
plt.show()
