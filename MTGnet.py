import torch
import torch.nn as nn
import torch.nn.functional as F
import pt_util
import matplotlib.pyplot as plt



class MTGNet(nn.Module):
    def __init__(self):
        super(MTGNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 256)
        self.lin1_dropout = nn.Dropout(.5)
        self.fc2 = nn.Linear(256, 6)
        self.accuracy = None

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(x)
        x = self.lin1_dropout(x)
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(x)
        x = self.lin1_dropout(x)
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.lin1_dropout(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if self.accuracy == None or accuracy > self.accuracy:
            self.accuracy = accuracy
            self.save_model(file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)


