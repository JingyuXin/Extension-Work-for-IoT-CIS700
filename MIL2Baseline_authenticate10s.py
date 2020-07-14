import SiameseNet
import random
import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.autograd import Variable

# Train a Siamese Network to compare 2 signals with length of 10s directly

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class bag:
    def __init__(self, Xpair_lst, label): # 1x3x200
        self.Xpairs = Xpair_lst
        self.label = label
    def getNumPairs(self):
        return len(self.Xpairs)

class x_pair:
    def __init__(self, x1, x2, label): # 1x3x200
        assert(x1.shape == x2.shape)
        self.x1 = x1
        self.x2 = x2
        self.label = label

def remove_col(df):
    return df.drop(['EID','time','time_in_ms'], axis =1)

# used to sample pure samples
def sample(path, file, length, start):
    df = pd.read_csv(path + file)
    df = remove_col(df)
    end = len(df)
    idx = np.random.randint(start, end)
    while(idx + length >= end):
        idx = np.random.randint(start, end)
    res = df.iloc[idx:idx+length].values
    return res

# extract a certain length signal from one in the files
def Extract(usr, path, files, size):
    extractFrom = np.random.randint(0,len(files))
    while(files[extractFrom][:-12] == usr):
        extractFrom = np.random.randint(0, len(files))
    df = pd.read_csv(path+files[extractFrom])
    start = np.random.randint(0,len(df)-size)
    x = df.iloc[start:start+size]
    x = remove_col(x)
    return x.values

def ExpMovingAverage(array, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(array, weights, mode='full')[:len(array)]
    a[:window] = a[window]
    return a

# smooth data samples being corrupted
# smoothing range is from -50 to 50 around inserted position using window size 10
def smooth(copy, start, window = 10):
    res = copy.copy()
    start = start - 50
    end = start + 50
    data_x = copy[start:end,0]
    data_y = copy[start:end,1]
    data_z = copy[start:end,2]
    smoothed_x = ExpMovingAverage(data_x, window)
    smoothed_y = ExpMovingAverage(data_y, window)
    smoothed_z = ExpMovingAverage(data_z, window)
    res[start + window +1 : end - window, 0] = smoothed_x[window+1:-window]
    res[start + window +1 : end - window, 1] = smoothed_y[window+1:-window]
    res[start + window +1 : end - window, 2] = smoothed_z[window+1:-window]
    return res


# given a pure sample, insert a portion from other user to create a synthetic sample
# size = 100, 200, 300, 400 and 500 in my experiments
def CreateSyn(usr, puredata, path, files, size):
    assert(len(puredata) > size)
    copy = puredata.copy()
    start = np.random.randint(200, len(puredata)-size-100)
    end = start + size
    # assume attack happens after at least 2s
    replace_clip = Extract(usr, path, files, size)
    copy[start:(start+size), :] = replace_clip
    return copy, start, end

# generate num_sample synthetic samples
def SyntheticGenerator(path, usr_file, files, num_sample, replace_size, puresize, start, ifSmooth = True):
    X = np.zeros((num_sample, puresize, 3))
    usr = usr_file[:-12]
    #print(usr_file)
    for i in range(num_sample):
        puredata = sample(path, usr_file, puresize, start)
        copy, smooth_start, smooth_end = CreateSyn(usr, puredata, path, files, replace_size)
        if(ifSmooth):
            copy = smooth(copy, smooth_start)
            copy = smooth(copy, smooth_end)
        X[i,:,:] = copy
    return X

# for a session, make 15 label 1 bags and 15 label 0 bags
def makePairs(path, usrSess, files, num = 10, pureSize = 1000, ifSmooth = True):
    df = pd.read_csv(path + usrSess)
    df = remove_col(df)

    startpoint = pureSize + 0

    template = df.iloc[0:startpoint].values

    Xpure = np.zeros((num+int(num/2), pureSize, 3))
    for i in range(num+int(num/2)): # extract pure parts
        x = sample(path, usrSess, pureSize, startpoint)
        Xpure[i,:,:] = x
    #print("Pure 10s extracted!")
    Xsyn100 = SyntheticGenerator(path, usrSess, files, int(num/5), 100, pureSize, startpoint, ifSmooth= ifSmooth)
    Xsyn200 = SyntheticGenerator(path, usrSess, files, int(num / 5), 200, pureSize, startpoint,  ifSmooth= ifSmooth)
    Xsyn300 = SyntheticGenerator(path, usrSess, files, int(num / 5), 300, pureSize, startpoint,  ifSmooth= ifSmooth)
    Xsyn400 = SyntheticGenerator(path, usrSess, files, int(num / 5), 400, pureSize, startpoint, ifSmooth= ifSmooth)
    Xsyn500 = SyntheticGenerator(path, usrSess, files, int(num / 5), 500, pureSize, startpoint,  ifSmooth= ifSmooth)
    Xsyn = np.vstack([Xsyn100, Xsyn200, Xsyn300, Xsyn400, Xsyn500])
    #print("Synthetic 10s generated!")

    imposter = np.zeros((int(num/2), pureSize, 3))
    usr = usrSess[:-12]
    for i in range(int(num/2)):
        imposter[i, :, :] = Extract(usr, path, files, pureSize)
    pairLst = []
    for i in range(int(1.5*num)):
        #print("Createing bag No. {}".format(i))
        pair = x_pair(template, Xpure[i,:,:], 1)
        #print("number of pairs in a posBag is {}".format(posBag.getNumPairs()))
        pairLst.append(pair)
    for i in range(num):
        pair = x_pair(template, Xsyn[i, :, :], 0)
        #print("number of pairs in a negBag is {}".format(negBag.getNumPairs()))
        pairLst.append(pair)
    for i in range(int(num/2)):
        pair = x_pair(template, imposter[i, :, :], 0)
        pairLst.append(pair)
    assert(len(pairLst) == 30)
    return pairLst

def makeGroupPairs(datapath, groupfiles, ifSmooth = True):
    pairslst = []
    for file in groupfiles:
        pairs = makePairs(datapath, file, groupfiles, ifSmooth= ifSmooth)
        pairslst += pairs
    print("Number of pairs is {}.".format(len(pairslst)))
    # pos = 0
    # neg = 0
    # for bag in baglst:
    #     if(bag.label == 1):
    #         pos += 1
    #     else:
    #         neg += 1
    # print(neg, pos)
    return pairslst

def takeBatch(batchsize, trainPairs):
    h,w = trainPairs[0].x1.shape # 1000 x 3
    num_sample = len(trainPairs)
    random.shuffle(trainPairs)
    for i in range(0, num_sample - batchsize + 1, batchsize):
        x1s = np.zeros((batchsize, w, h))
        x2s = np.zeros((batchsize, w, h))
        ys = np.zeros((batchsize, 1))
        for j in range(batchsize):
            x1s[j, :, :] = trainPairs[i + j].x1.T
            x2s[j, :, :] = trainPairs[i + j].x2.T
            ys[j] = trainPairs[i + j].label

        x1s = torch.from_numpy(x1s).float()
        x2s = torch.from_numpy(x2s).float()
        x1s = x1s.view(batchsize, 1, w, h)
        x2s = x2s.view(batchsize, 1, w, h)
        yield x1s, x2s, ys

def test(xpairs, net):
    h, w = xpairs[0].x1.shape
    num_sample = len(xpairs)
    x1s = np.zeros((num_sample, w, h))
    x2s = np.zeros((num_sample, w, h))
    ys = np.zeros((num_sample, 1))

    for i in range(num_sample):
        x1s[i, :, :] = xpairs[i].x1.T
        x2s[i, :, :] = xpairs[i].x2.T
        ys[i] = xpairs[i].label
    x1s = torch.from_numpy(x1s).float()
    x2s = torch.from_numpy(x2s).float()
    x1s = x1s.view(num_sample, 1, w, h)
    x2s = x2s.view(num_sample, 1, w, h)
    ys = torch.from_numpy(ys)

    FN, FP, TP, TN, correct = 0, 0, 0, 0, 0
    num_pos = 0
    num_neg = 0

    if (torch.cuda.is_available()):
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
                num_pos += 1
                TP += 1
                correct += 1
            else:
                num_neg += 1
                FP += 1
        else:
            if (ys[i] == 0):
                num_neg += 1
                TN += 1
                correct += 1
            else:
                num_pos += 1
                FN += 1
    if (TP == 0):
        precision, recall, F1 = 0, 0, 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
    acc = correct / num_sample
    print("Evaluation result: TP {}/{}, TN {}/{}, F1: {}".format(TP, num_pos, TN, num_neg, F1))
    return F1, precision, recall, acc

def trainBatch(net, trainFiles, valPairs, outdir, batch_size=10, epochs=200, lr=0.0005):
    val_F1 = 0
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    loss_func = nn.BCELoss()

    if (torch.cuda.is_available()):
        #net = nn.DataParallel(net)
        net = net.cuda()
        loss_func = loss_func.cuda()
    for epoch in range(epochs):
        trainPairs = makeGroupPairs(datapath, trainFiles, ifSmooth=ifSmooth)
        train_loss = []
        net.train()
        for x1s, x2s, ys in takeBatch(batch_size, trainPairs):
            targets = torch.from_numpy(ys).view(batch_size, -1)
            if(torch.cuda.is_available()):
                targets = targets.cuda()
                x1s = x1s.cuda()
                x2s = x2s.cuda()
            x1s,x2s, targets = Variable(x1s), Variable(x2s), Variable(targets.float())
            opt.zero_grad()

            out = net(x1s, x2s)
            loss = loss_func(out, targets)
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
        F1, precision, recall, acc = test(valPairs, net)

        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)),
              "Validation F1: {:.4f}...".format(F1),
              "Precision and Recall on Validation Set: {:.4f}, {:.4f}".format(precision, recall))

        if (F1 >= val_F1):  # store the model performs best on validation set
            best_model = net.state_dict()
            val_F1 = F1
            torch.save(best_model, outdir + 'Best_model.pth')
            print('saving a best model...')

        final_model = net.state_dict()
        if ((epoch + 1) % 10 == 0):
            torch.save(final_model, outdir + 'Final_model.pth')
    return final_model, best_model


if __name__ == '__main__':
    datapath = "/home/jxin05/rawdata/"
    modelpath = '/home/jxin05/10sPhase1/'
    outdir = '/home/jxin05/10sPhase1/'
    ifSmooth = True
    f = open('/home/jxin05/Phase1Data/testFiles.txt', 'r')
    testFiles = f.read()
    testFiles = eval(testFiles)
    f.close()

    f = open('/home/jxin05/Phase1Data/validationFiles.txt', 'r')
    valFiles = f.read()
    valFiles = eval(valFiles)
    f.close()

    f = open('/home/jxin05/Phase1Data/trainFiles.txt', 'r')
    trainFiles = f.read()
    trainFiles = eval(trainFiles)
    f.close()

    #trainPairs = makeGroupPairs(datapath, trainFiles,ifSmooth=ifSmooth)
    valPairs = makeGroupPairs(datapath, valFiles, ifSmooth=ifSmooth)
    #testPairs = makeGroupPairs(datapath, testFiles, ifSmooth=ifSmooth)

    net = SiameseNet.SiameseNetwork10s()
    net.load_state_dict(torch.load(modelpath + 'phase1_best_model.pth'))
    #net.apply(SiameseNet.init_weights)

    final_state, best_state = trainBatch(net, trainFiles, valPairs, outdir, epochs=200)

    best_model = SiameseNet.SiameseNetwork10s()
    best_model.load_state_dict(best_state)

    print("Evaluate on Test set...")
    for i in range(10):
        testPairs = makeGroupPairs(datapath, testFiles, ifSmooth=ifSmooth)
        test(testPairs, best_model)