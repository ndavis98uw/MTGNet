import torch
from PIL import Image
import numpy as np
import pickle
import torchvision.transforms.functional as TF
import torchvision.transforms
import random
TRAIN_VAL_SPLIT = .8


class MTGDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels, ids, indices, is_color, label_dic, run_type = 0):
        super(MTGDataset, self).__init__()
        self.is_color = is_color
        self.data_dir = data_dir
        self.is_train = run_type == 0
        self.ids = np.load(ids)
        self.labels = np.load(labels)
        if run_type == 0:
            train_indices = indices[:int(len(indices)*TRAIN_VAL_SPLIT)]
            self.ids = self.ids[train_indices]
            self.labels = self.labels[train_indices]
        elif run_type == 1:
            test_indices = indices[int(len(indices)*TRAIN_VAL_SPLIT):]
            self.ids = self.ids[test_indices]
            self.labels = self.labels[test_indices]
        with open(label_dic.split('.')[0] + '.pickle', 'rb') as handle:
            self.label_dict = pickle.load(handle)

    def __len__(self):
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
        if self.is_train:
            angle = random.randrange(-15, 16, 2)
            image = TF.rotate(image, angle)
            jitter = torchvision.transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1)
            image = jitter.forward(image)
        if not self.is_color:
            label = self.label_dict[self.labels[idx]]
        else:
            label = self.labels[idx]
        return image, label


def data_fetcher(index_dirs, data_dir, is_color=True):
    if is_color:
        label_type = 'colors'
    else:
        label_type = 'types'
    ids = np.load(index_dirs + '/ids_train.npy')
    indices = np.arange(ids.shape[0])
    np.random.shuffle(indices)
    train_dataset = MTGDataset(data_dir, index_dirs + '/' + label_type + "_train" + '.npy', index_dirs + '/ids_train.npy', indices, is_color, index_dirs + '/' + label_type + '.npy', run_type=0)
    val_dataset = MTGDataset(data_dir, index_dirs + '/' + label_type + "_train" + '.npy', index_dirs + '/ids_train.npy', indices, is_color, index_dirs + '/' + label_type + '.npy',  run_type=1)
    class_names = []
    with open((index_dirs + '/' + label_type + '.npy').split('.')[0] + '.pickle', 'rb') as handle:
        label_dict = pickle.load(handle)
    class_names = list(label_dict.keys())
    return class_names, train_dataset, val_dataset


def test_fetcher(index_dirs, data_dir, is_color=True):
    if is_color:
        label_type = 'colors'
    else:
        label_type = 'types'
    ids = np.load(index_dirs + '/ids_test.npy')
    indices = np.arange(ids.shape[0])
    np.random.shuffle(indices)
    test_dataset = MTGDataset(data_dir, index_dirs + '/' + label_type + '_test' + '.npy', index_dirs + '/ids_test.npy', indices, is_color, index_dirs + '/' + label_type + '.npy', run_type=2)
    return test_dataset
