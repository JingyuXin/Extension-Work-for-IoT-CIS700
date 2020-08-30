import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 1

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
        #loss = (torch.min(y_pred, 1)[0] - label)**2
        lossfunc = torch.nn.BCELoss()
        loss = lossfunc(torch.min(y_pred, 1)[0].double(), label.double())
        #loss = torch.mean(-label*torch.log(torch.min(y_pred, 1)[0])-(1-label)*torch.log(1-torch.min(y_pred, 1)[0]))
        return loss

class SiameseNetwork2s(nn.Module):
    def __init__(self):
        super(SiameseNetwork2s, self).__init__()

        self.cnn1 = nn.Sequential(nn.Conv2d(1, 256, (3, 3), stride=(1, 2)),
                                  nn.LeakyReLU(inplace=True),
                                  nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.cnn2 = nn.Sequential(nn.Conv1d(256, 128, 4, stride=2),
                                  nn.LeakyReLU(inplace=True),
                                  nn.MaxPool1d(3, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(1408, 512),
                                 nn.Dropout(0.3),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(512, 256),
                                 nn.Dropout(0.3),
                                 nn.LeakyReLU(inplace=True)
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
        f = self.fc2(dist)
        #f = dist
        return f


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 1
        self.D = 64
        self.K = 1

        self.Siamese = SiameseNetwork2s()

        self.attention = nn.Sequential(nn.Linear(self.L, self.D),
                                       nn.Tanh(),
                                       nn.Linear(self.D, self.K))

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1),
                                        nn.Sigmoid())

    def forward(self, x1s, x2s):  # (self, x1s, x2s)

        H = self.Siamese(x1s, x2s)

        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob.cuda(), Y_hat.cuda(), A.cuda()

    def calculate_objective(self, x1s, x2s, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(x1s, x2s)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 1
        self.D = 64
        self.K = 1

        self.Siamese = SiameseNetwork2s()

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x1s, x2s):  # (self, x1s, x2s)

        H = self.Siamese(x1s, x2s)
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A