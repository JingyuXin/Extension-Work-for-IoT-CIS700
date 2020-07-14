import SiameseNet
import random
import numpy as np
import torch
import pandas as pd

# Evaluate a MIL-Siamese model to authenticate different categories of data
# Pure data from genuine user
# Pure data from imposters
# Synthetic data (1s, 2s, 3s, 4s, 5s replaced)

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

def remove_col(df):
    return df.drop(['EID', 'time', 'time_in_ms'], axis=1)

def sample(path, file, length, start):
    df = pd.read_csv(path + file)
    df = remove_col(df)
    end = len(df)
    idx = np.random.randint(start, end)
    while(idx + length >= end):
        idx = np.random.randint(start, end)
    res = df.iloc[idx:idx+length].values
    return res

def Extract(usr, path, files, size):
    extractFrom = np.random.randint(0,len(files))
    while (files[extractFrom][:-12] == usr):
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

def CreateSyn(usr, puredata, path, files, size):
    assert(len(puredata) > size)
    copy = puredata.copy()
    start = np.random.randint(100, len(puredata)-size-100)
    end = start + size
    # assume attack happens after at least 1s
    replace_clip = Extract(usr, path, files, size)
    copy[start:(start+size), :] = replace_clip
    return copy, start, end

def SyntheticGenerator(path, usr_file, files, num_sample, replace_size, puresize, start, ifSmooth = True):
    X = np.zeros((num_sample, puresize, 3))
    usr = usr_file[:-12]
    for i in range(num_sample):
        puredata = sample(path, usr_file, puresize, start)
        copy, smooth_start, smooth_end = CreateSyn(usr, puredata, path, files, replace_size)
        if(ifSmooth):
            copy = smooth(copy, smooth_start)
            copy = smooth(copy, smooth_end)
        X[i,:,:] = copy
    return X

def ImposterGenerator(path, usr_file, files, num_each, puresize, start):
    usr = usr_file[:-12]
    X = []
    for file in files:
        if(file[:-12] == usr):
            continue
        for i in range(num_each):
            imposterdata = sample(path, file, puresize, start)
            imposterdata = imposterdata[np.newaxis, :]
            X.append(imposterdata)
    X = np.vstack(X)
    return X

def take_batch(batch_size, groupBag):
    indices = np.arange(len(groupBag))
    np.random.shuffle(indices)
    for i in range(0,len(groupBag)-batch_size+1, batch_size):
        excerpt = indices[i:i + batch_size]
        res = []
        for idx in excerpt:
            res.append(groupBag[idx])
        yield res


def makePairs(X1, X2, numPairs, size = 200): # X1, X2 : 1000x3
    res = []
    for i in range(numPairs):
        idx1 = np.random.randint(0, X1.shape[0])
        while(idx1 + size >= X1.shape[0]):
            idx1 = np.random.randint(0, X1.shape[0])
        idx2 = np.random.randint(0, X2.shape[0])
        while (idx2 + size >= X2.shape[0]):
            idx2 = np.random.randint(0, X2.shape[0])

        x1 = X1[idx1:idx1+size, :]
        x2 = X2[idx2:idx2+size, :]

        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        x1 = x1.view(1, 1, 3, -1)
        x2 = x2.view(1, 1, 3, -1)
        res.append([x1,x2])
    return res

def testPerformance(groupBag, net):
    num_sample = len(groupBag)
    FN, FP, TP, TN, correct = 0, 0, 0, 0, 0
    num_pos = 0
    num_neg = 0
    for aBag in groupBag:
        y_pred = []
        for pair in aBag.Xpairs:
            x1 = pair[0]
            x2 = pair[1]
            if (torch.cuda.is_available()):
                x1 = x1.cuda()
                x2 = x2.cuda()
                net = net.cuda()
            with torch.no_grad():
                y_pred.append(net(x1, x2).item())
        score = min(y_pred)
        if (score > 0.5):
            if (aBag.label == 1):
                correct += 1
                TP += 1
                num_pos += 1
            else:
                num_neg += 1
                FP += 1
        else:
            if (aBag.label == 0):
                correct += 1
                TN += 1
                num_neg += 1
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



if __name__ == '__main__':
    outdir = '/home/jxin05/newCode/MIL2_imposter/10s/'
    #modelpath = "/home/jxin05/newCode/MIL2/"
    modelpath = '/home/jxin05/newCode/MIL2_authentication/10s/'
    datapath = "/home/jxin05/rawdata/"
    ifSmooth = True
    numpairs = 8
    print("This experiment is for personalization using MIL2, ifSmooth = {}, numpairs = {}. Train on session 1 and test on session 2".format(ifSmooth, numpairs))
    # load data
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

    net = SiameseNet.ComplexSiameseNet()
    net.load_state_dict(torch.load(modelpath + 'best_MILmodel.pth'))

    templatesize = 1000
    num_sample = 15
    pureSize = 1000
    res = {}

    for file in testFiles:
        print("Evaluating using {}".format(file))
        res[file] = {'accPure': [], 'acc100': [], 'acc200': [], 'acc300': [],
                     'acc400': [], 'acc500': [], 'accImposter': []}
        usr = file[:-12]
        df = pd.read_csv(datapath + file)
        df = remove_col(df)
        template = df.iloc[0:templatesize].values
        Xpure = np.zeros((num_sample, pureSize, 3))
        Ximposter = np.zeros((num_sample, pureSize, 3))

        for j in range(10):
            for i in range(num_sample):
                Xpure[i, :, :] = sample(datapath, file, pureSize, templatesize)
                Ximposter[i, :, :] = Extract(usr, datapath, testFiles, pureSize)
            Xsyn100 = SyntheticGenerator(datapath, file, testFiles, num_sample, 100, pureSize, templatesize,
                                         ifSmooth=ifSmooth)
            Xsyn200 = SyntheticGenerator(datapath, file, testFiles, num_sample, 200, pureSize, templatesize,
                                         ifSmooth=ifSmooth)
            Xsyn300 = SyntheticGenerator(datapath, file, testFiles, num_sample, 300, pureSize, templatesize,
                                         ifSmooth=ifSmooth)
            Xsyn400 = SyntheticGenerator(datapath, file, testFiles, num_sample, 400, pureSize, templatesize,
                                         ifSmooth=ifSmooth)
            Xsyn500 = SyntheticGenerator(datapath, file, testFiles, num_sample, 500, pureSize, templatesize,
                                         ifSmooth=ifSmooth)
            purelst, syn100lst, syn200lst, syn300lst, syn400lst, syn500lst, imposterlst = [], [], [], [], [], [], []
            for i in range(num_sample):
                pairs = makePairs(template, Xpure[i, :, :], numpairs)
                purelst.append(bag(pairs, 1))
                pairs = makePairs(template, Xsyn100[i, :, :], numpairs)
                syn100lst.append(bag(pairs, 0))
                pairs = makePairs(template, Xsyn200[i, :, :], numpairs)
                syn200lst.append(bag(pairs, 0))
                pairs = makePairs(template, Xsyn300[i, :, :], numpairs)
                syn300lst.append(bag(pairs, 0))
                pairs = makePairs(template, Xsyn400[i, :, :], numpairs)
                syn400lst.append(bag(pairs, 0))
                pairs = makePairs(template, Xsyn500[i, :, :], numpairs)
                syn500lst.append(bag(pairs, 0))
                pairs = makePairs(template, Ximposter[i, :, :], numpairs)
                imposterlst.append(bag(pairs, 0))
            accPure = testPerformance(purelst, net)[3]
            acc100 = testPerformance(syn100lst, net)[3]
            acc200 = testPerformance(syn200lst, net)[3]
            acc300 = testPerformance(syn300lst, net)[3]
            acc400 = testPerformance(syn400lst, net)[3]
            acc500 = testPerformance(syn500lst, net)[3]
            accImposter = testPerformance(imposterlst, net)[3]

            res[file]['accPure'].append(accPure)
            res[file]['acc100'].append(acc100)
            res[file]['acc200'].append(acc200)
            res[file]['acc300'].append(acc300)
            res[file]['acc400'].append(acc400)
            res[file]['acc500'].append(acc500)
            res[file]['accImposter'].append(accImposter)

        f = open(modelpath + 'CategoricalRes.txt', 'w')
        f.write(str(res))
        f.close()
        







