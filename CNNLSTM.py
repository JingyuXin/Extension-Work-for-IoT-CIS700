import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random

# Structure of CNN-LSTM model

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def transformation(X, overlap = 0.2, stepsize = 100):
    if(len(X.shape) != 3):
        X = np.array([X])
    start = 0
    length = X.shape[1]
    num_step = 0
    x_lst = []
    while(start + stepsize <= length - 100):
        x = X[:, start:start + stepsize, :]
        start += int((1-overlap)*stepsize)
        x_lst.append(x)
        num_step += 1
    x_lst.append(X[:, 900:, :])
    num_step += 1
    num_sample = X.shape[0]
    newX = np.zeros((num_sample, num_step, stepsize, 3))
    for i in range(num_step):
        newX[:,i,:,:] = x_lst[i]
    res = torch.from_numpy(newX).float()
    res = res.transpose(2,3)
    res = res.view(num_sample, num_step, 1, 3, stepsize)
    if(torch.cuda.is_available()):
        res = res.cuda()
    return res


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(1, 256, (3, 3), stride=(1, 2)),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(256),
                                  nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.cnn2 = nn.Sequential(nn.Conv1d(256, 128, 4, stride=2),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm1d(128),
                                  nn.MaxPool1d(3, stride=2))

    def forward(self, x):
        output = self.cnn1(x)
        output = output.squeeze(2)
        output = self.cnn2(output)

        return output


class cnn_lstm(nn.Module):
    def __init__(self, cnn=None, drop_prob=0.4, n_class=2, n_layer=2):
        super(cnn_lstm, self).__init__()
        if (cnn == None):
            self.cnn = CNN()
        else:
            self.cnn = cnn
        self.n_class = n_class
        self.lstm = nn.LSTM(640, 256, num_layers=n_layer, batch_first=True)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_class)
        self.dropout = nn.Dropout(drop_prob)
        self.outlayer = nn.Softmax(dim=2)

    def forward(self, x, hidden):
        inputs = transformation(x)
        batch_size, timesteps, C, H, W = inputs.size()
        inputs = inputs.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(inputs)
        lstm_in = c_out.view(batch_size, timesteps, -1)
        lstm_out, hidden = self.lstm(lstm_in, hidden)
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.outlayer(out)

        if (batch_size != 1):
            out = out.transpose(1, 2)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(2, batch_size, 256).zero_(),
                  weight.new(2, batch_size, 256).zero_())
        if(torch.cuda.is_available()):
            hidden = (weight.new(2, batch_size, 256).zero_().cuda(),
                  weight.new(2, batch_size, 256).zero_().cuda())
        return hidden


def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv2d or type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)