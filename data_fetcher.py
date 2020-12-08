import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import pickle
TRAIN_TEST_SPLIT = .8


class MTGDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels, ids, indices, is_train=True):
        super(MTGDataset, self).__init__()
        self.data_dir = data_dir
        self.ids = np.load(ids)
        self.labels = np.load(labels)
        if is_train:
            train_indices = indices[:int(len(indices)*TRAIN_TEST_SPLIT)]
            self.ids = self.ids[train_indices]
            self.labels = self.labels[train_indices]
        else:
            test_indices = indices[int(len(indices)*TRAIN_TEST_SPLIT):]
            self.ids = self.ids[test_indices]
            self.labels = self.labels[test_indices]
        with open(labels.split('.')[0] + '.pickle', 'rb') as handle:
            self.label_dict = pickle.load(handle)

    def __len__(self):
        # TODO return the number of unique sequences you have, not the number of characters.
        return len(self.ids)

    def __getitem__(self, idx):
        # Return the data and label for a character sequence as described above.
        # The data and labels should be torch long tensors.
        # You should return a single entry for the batch using the idx to decide which chunk you are
        # in and how far down in the chunk you are.
        image = Image.open(self.data_dir + '/' + self.ids[idx] + '.png')
        image = image.convert('RGB')
        image = np.asarray(image).transpose(-1, 0, 1)
        image = image/255
        image = torch.from_numpy(np.asarray(image))
        image = image.float()
        label = self.label_dict[self.labels[idx]]
        return image, label


def data_fetcher(index_dirs, data_dir, label_type):
    ids = np.load(index_dirs + '/ids.npy')
    labels = np.load(index_dirs + '/' + label_type + '.npy')
    indices = np.arange(ids.shape[0])
    np.random.shuffle(indices)
    train_dataset = MTGDataset(data_dir, index_dirs + '/' + label_type + '.npy', index_dirs + '/ids.npy', indices, is_train=True)
    test_dataset = MTGDataset(data_dir, index_dirs + '/' + label_type + '.npy', index_dirs + '/ids.npy', indices, is_train=False)
    class_names = []
    with open((index_dirs + '/' + label_type + '.npy').split('.')[0] + '.pickle', 'rb') as handle:
        label_dict = pickle.load(handle)
    class_names = list(label_dict.keys())
    return class_names, train_dataset, test_dataset
