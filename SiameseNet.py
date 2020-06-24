import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def L1_distance(y1, y2):
    num_sample = y1.size()[0]
    out = torch.zeros((num_sample, 1))
    for i in range(num_sample):
        dist = torch.dist(y1[i], y2[i], 1)
        out[i] = dist
    return out

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

class bagLoss(nn.Module):
    def __init__(self):
        super(bagLoss, self).__init__()

    def forward(self, y_pred, label): # y_pred: 5x1, label: 1x1
        if (torch.cuda.is_available()):
            y_pred = y_pred.cpu()
            label = label.cpu()
        loss = (torch.min(y_pred, 1)[0] - label)**2
        return torch.mean(loss)

class ComplexSiameseNet(nn.Module):
    def __init__(self):
        super(ComplexSiameseNet, self).__init__()

        self.cnn1 = nn.Sequential(nn.Conv2d(1, 256, (3, 3), stride=(1, 2)),
                                  #nn.ReLU(inplace=True),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(256),
                                  nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.cnn2 = nn.Sequential(nn.Conv1d(256, 128, 4, stride=2),
                                  nn.LeakyReLU(inplace=True),
                                  #nn.ReLU(inplace=True),
                                  nn.BatchNorm1d(128),
                                  nn.MaxPool1d(3, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(1408, 512),
                                 nn.Dropout(0.4),
                                 nn.LeakyReLU(inplace=True),
                                 #nn.ReLU(inplace=True),
                                 nn.Linear(512, 256),
                                 nn.Dropout(0.4),
                                 nn.LeakyReLU(inplace=True)
                                 #nn.ReLU(inplace=True),
                                 #nn.Linear(256, 128),
                                 #nn.Dropout(0.4)
                                 )
        self.fc2 = nn.Sequential(nn.Linear(1, 1),
                                 nn.Sigmoid())

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.squeeze(2)
        output = self.cnn2(output)
        output = output.view(output.size(0), -1)
        y = self.fc1(output)
        return y

    def forward(self, x1, x2):
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        dist = L1_distance(y1, y2)
        if(torch.cuda.is_available()):
            dist = dist.cuda()
        # print(dist.size())
        f = self.fc2(dist)
        return f

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(nn.Conv2d(1, 128, (3, 3), stride=(1, 2)),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm2d(128),
                                  nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.cnn2 = nn.Sequential(nn.Conv1d(128, 64, 4, stride=2),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm1d(64),
                                  nn.MaxPool1d(3, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(704, 256),
                                 nn.Dropout(0.3),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256, 128),
                                 nn.Dropout(0.3),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128, 64),
                                 nn.Dropout(0.3)
                                 )
        self.fc2 = nn.Sequential(nn.Linear(1, 1),
                                 nn.Sigmoid())

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.squeeze(2)
        output = self.cnn2(output)
        output = output.view(output.size(0), -1)
        y = self.fc1(output)
        return y

    def forward(self, x1, x2):
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        dist = L1_distance(y1, y2)
        if(torch.cuda.is_available()):
            dist = dist.cuda()
        # print(dist.size())
        f = self.fc2(dist)
        return f