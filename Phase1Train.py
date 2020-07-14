import SiameseNet
import pickle
import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# Train a Siamese Network to distinguish pure data from different users
# helps converaging of phase 2 training

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

datapath = "/home/jxin05/rawdata/"

class x_pair:
    def __init__(self, x1, x2, label): # 1x3x200
        assert(x1.shape == x2.shape)
        self.x1 = x1
        self.x2 = x2
        self.label = label

def takeBatch(batchsize, trainSet, h, w):
    num_sample = len(trainSet)
    random.shuffle(trainSet)
    for i in range(0, num_sample - batchsize + 1, batchsize):
        x1s = np.zeros((batchsize, h, w))
        x2s = np.zeros((batchsize, h, w))
        ys = np.zeros((batchsize, 1))
        for j in range(batchsize):
            x1s[j, :, :] = trainSet[i + j].x1
            x2s[j, :, :] = trainSet[i + j].x2
            ys[j] = trainSet[i + j].label

        x1s = torch.from_numpy(x1s).float()
        x2s = torch.from_numpy(x2s).float()
        x1s = x1s.view(batchsize, 1, h, w)
        x2s = x2s.view(batchsize, 1, h, w)
        yield x1s, x2s, ys


def test(X_list, net):
    h, w = X_list[0].x1.shape
    num_sample = len(X_list)
    x1s = np.zeros((num_sample, h, w))
    x2s = np.zeros((num_sample, h, w))
    ys = np.zeros((num_sample, 1))

    for i in range(num_sample):
        x1s[i, :, :] = X_list[i].x1
        x2s[i, :, :] = X_list[i].x2
        ys[i] = X_list[i].label
    x1s = torch.from_numpy(x1s).float()
    x2s = torch.from_numpy(x2s).float()
    x1s = x1s.view(num_sample, 1, h, w)
    x2s = x2s.view(num_sample, 1, h, w)
    ys = torch.from_numpy(ys)

    if(torch.cuda.is_available()):
        x1s = x1s.cuda()
        x2s = x2s.cuda()
        net = net.cuda()
    net.eval()
    with torch.no_grad():
        s = net(x1s, x2s)

    correct = 0
    for i in range(num_sample):
        if (s[i] > 0.5):
            if (ys[i] == 1):
                correct += 1
        else:
            if (ys[i] == 0):
                correct += 1

    acc = correct / num_sample
    return acc


def trainBatch(net, trainDict, valDict, outdir, batch_size=10, epochs=200, lr=0.0005):
    num_sample = len(trainDict['same']) + len(trainDict['diff'])
    h, w = trainDict['same'][0].x1.shape
    dataset = trainDict['same'] + trainDict['diff']
    random.shuffle(dataset)
    trainSet = dataset


    val_acc = 0
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    # opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    loss_func = nn.BCELoss()

    if (torch.cuda.is_available()):
        #net = nn.DataParallel(net)
        net = net.cuda()
        loss_func = loss_func.cuda()
    for epoch in range(epochs):
        train_loss = []
        net.train()
        # count = 0
        for x1s, x2s, ys in takeBatch(batch_size, trainSet, h, w):
            targets = torch.from_numpy(ys).view(batch_size, -1)
            # targets = targets.squeeze()
            if(torch.cuda.is_available()):
                targets = targets.cuda()
                x1s = x1s.cuda()
                x2s = x2s.cuda()
            x1s,x2s, targets = Variable(x1s), Variable(x2s), Variable(targets.float())
            opt.zero_grad()

            out = net(x1s, x2s)
            #             print(out.size(), targets.size())
            #             print(out)
            #             print(targets)
            loss = loss_func(out, targets)
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
        acc1 = test(valDict['same'], net)
        acc2 = test(valDict['diff'], net)
        acc = (acc1 + acc2) / 2

        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)),
              "Validation TNR: {:.4f}...".format(acc2),
              "Validation TPR: {:.4f}...".format(acc1),
              "Validation acc: {:.4f}...".format(acc))

        if (acc >= val_acc):  # store the model performs best on validation set
            best_model = net.state_dict()
            val_acc = acc
            torch.save(best_model, outdir + 'phase1_best_model.pth')
            print('saving a best model...')

        final_model = net.state_dict()
        if ((epoch + 1) % 10 == 0):
            torch.save(final_model, outdir + 'phase1_final_model.pth')
    return final_model, best_model

if __name__ == '__main__':

    outdir = '/home/jxin05/10sPhase1/'
    data = '/home/jxin05/10sPhase1/'

    # load data
    f = open('/home/jxin05/Phase1Data/testFiles.txt', 'r')
    testFiles = f.read()
    testFiles = eval(testFiles)
    f.close()

    f = open('/home/jxin05/Phase1Data/trainFiles.txt', 'r')
    trainFiles = f.read()
    trainFiles = eval(trainFiles)
    f.close()

    f = open('/home/jxin05/Phase1Data/validationFiles.txt', 'r')
    valFiles = f.read()
    valFiles = eval(valFiles)
    f.close()


    df = open(data + 'train.pickle', 'rb')
    X_train = pickle.load(df)
    df.close()

    df = open(data+'test.pickle', 'rb')
    X_test = pickle.load(df)
    df.close()

    df = open(data+'validation.pickle', 'rb')
    X_val = pickle.load(df)
    df.close()

    # net = SiameseNet.ComplexSiameseNet()
    # net.apply(SiameseNet.init_weights)
    net = SiameseNet.SiameseNetwork10s()
    net.apply(SiameseNet.init_weights)
    # phase 1: train on pairs
    print("Start Phase 1 training...")

    final_state, best_state = trainBatch(net, X_train, X_val, outdir,epochs = 200)
    best_model = SiameseNet.SiameseNetwork10s()
    #best_model = nn.DataParallel(best_model)
    best_model.load_state_dict(best_state)

    print("Finish Phase 1 training!")

    acc1 = test(X_test['same'], net)
    acc2 = test(X_test['diff'], net)
    acc = (acc1 + acc2) / 2

    print(
          "Testing TNR: {:.4f}...".format(acc2),
          "Testing TPR: {:.4f}...".format(acc1),
          "Testing acc: {:.4f}...".format(acc))