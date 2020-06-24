import SiameseNet
import SiameseNet2
import pickle
import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import pandas as pd

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

# used for generating batches for phase 1 training
# x1s[i] and x2s[i] form a pair
# under our setting, h = 3, w = 200
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

# used in phase 1 training to evaluate model on validation set
# X_list contains all label 1 pairs or all label 0 pairs
# X_list[i] is an instance of X_pair data structure
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

# training a model in phase 1
# store model parameters which works best on validation set
# store model after all epochs
def trainBatch(net, trainDict, valDict, batch_size=10, epochs=200, lr=0.0005):
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
            loss = loss_func(out, targets)
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
        acc1 = test(valDict['same'], net)
        acc2 = test(valDict['diff'], net)
        acc = (acc1 + acc2) / 2

        if (acc >= val_acc):  # store the model performs best on validation set
            best_model = net.state_dict()
            val_acc = acc
            torch.save(best_model, 'simple_best_model.pth')
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)),
              "Validation TNR: {:.4f}...".format(acc2),
              "Validation TPR: {:.4f}...".format(acc1),
              "Validation acc: {:.4f}...".format(acc))

        final_model = net.state_dict()
        if ((epoch + 1) % 10 == 0):
            torch.save(final_model, 'simple_final_model.pth')
    return final_model, best_model

def remove_col(df):
    return df.drop(['EID','time','time_in_ms'], axis =1)

def sample(file, length):
    df = pd.read_csv(file)
    df = remove_col(df)
    idx = np.random.randint(0, len(df))
    while(idx + length >= len(df)):
        idx = np.random.randint(0, len(df))
    res = df.iloc[idx:idx+length].values
    return res

# used for generating data for phase 2 training
# given a list of files in a group
# randomly pick a file and extract a portion with some size
def Extract(path, files, size):
    extractFrom = np.random.randint(0,len(files))
    df = pd.read_csv(path+files[extractFrom])
    start = np.random.randint(0,len(df)-size)
    x = df.iloc[start:start+size]
    x = remove_col(x)
    return x.values

# used for generating data for phase 2 training
# extract num pure data samples from a group
# size = 1000 in our setting
def ExtractPure(path, files, size, num):
    samples = np.zeros((num, size, 3))
    for i in range(num):
        x = Extract(path, files, size)
        samples[i,:,:] = x
    return samples


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
def CreateSyn(puredata, path, files, size):
    assert(len(puredata) > size)
    copy = puredata.copy()
    start = np.random.randint(200, len(puredata)-size-100)
    end = start + size
    # assume attack happens after at least 2s
    replace_clip = Extract(path, files, size)
    copy[start:(start+size), :] = replace_clip
    return copy, start, end

# generate num_sample synthetic samples
def SyntheticGenerator(path, files, num_sample, replace_size, puresize):
    X = np.zeros((num_sample, puresize, 3))
    for i in range(num_sample):
        puredata = Extract(path, files, puresize)
        copy, start, end = CreateSyn(puredata, path, files, replace_size)
        copy = smooth(copy, start)
        copy = smooth(copy, end)
        X[i,:,:] = copy
    return X

def DataGeneratePhase2(datapath, trainFiles, valFiles):
    XtrainPure = ExtractPure(datapath, trainFiles, 1000, 1000)
    Xtrain_100 = SyntheticGenerator(datapath, trainFiles, 300, 100, 1000)
    Xtrain_200 = SyntheticGenerator(datapath, trainFiles, 300, 200, 1000)
    Xtrain_300 = SyntheticGenerator(datapath, trainFiles, 300, 300, 1000)
    Xtrain_400 = SyntheticGenerator(datapath, trainFiles, 300, 400, 1000)
    Xtrain_500 = SyntheticGenerator(datapath, trainFiles, 300, 500, 1000)

    Xval_100 = SyntheticGenerator(datapath, valFiles, 100, 100, 1000)
    Xval_200 = SyntheticGenerator(datapath, valFiles, 100, 200, 1000)
    Xval_300 = SyntheticGenerator(datapath, valFiles, 100, 300, 1000)
    Xval_400 = SyntheticGenerator(datapath, valFiles, 100, 400, 1000)
    Xval_500 = SyntheticGenerator(datapath, valFiles, 100, 500, 1000)
    XvalPure = ExtractPure(datapath, valFiles, 1000, 100)

    YvalPure = np.ones(XvalPure.shape[0])
    YvalSyn = np.zeros(Xval_100.shape[0] * 5)
    Yval = np.hstack([YvalPure, YvalSyn])
    Xval = np.vstack([XvalPure, Xval_100, Xval_200, Xval_300, Xval_400, Xval_500])

    YtrainPure = np.ones(XtrainPure.shape[0])
    YtrainSyn = np.zeros(Xtrain_100.shape[0] * 5)
    Ytrain = np.hstack([YtrainPure, YtrainSyn])
    Xtrain = np.vstack([XtrainPure, Xtrain_100, Xtrain_200, Xtrain_300, Xtrain_400, Xtrain_500])

    return Xtrain, Ytrain, Xval, Yval

# evaluate phase 2 model
# num_pairs is number of pairs will be extracted from a 10s data sample
def testPerformance(net, X, Y, num_pairs):
    num_sample = X.shape[0]
    correct = 0
    FN = 0
    FP = 0
    TP = 0
    TN = 0
    for i in range(num_sample):
        pairs = SampleMI(X[i], num_pairs)
        y_pred = []
        for j in range(len(pairs)):
            x1s = pairs[j][0]
            x2s = pairs[j][1]
            if (torch.cuda.is_available()):
                x1s = x1s.cuda()
                x2s = x2s.cuda()
                net = net.cuda()
            with torch.no_grad():
                y_pred.append(net(x1s, x2s).item())
        score = min(y_pred) # label is determined by the pair having the lowest score
        #print(score)
        if(score >0.5):
            if(Y[i] == 1):
                correct += 1
                TP += 1
            else:
                FP += 1
        else:
            if(Y[i] == 0):
                correct += 1
                TN += 1
            else:
                FN += 1
    if(TP == 0):
        precision, recall, F1 = 0, 0, 0
    else:
        precision = TP/(TP + FP)
        recall = TP/(TP+FN)
        F1 = 2*precision*recall/(precision+recall)
    acc = correct/num_sample
    return F1, precision, recall, acc

# X: 1000 x3
# given a 10s data sample, extract num_pairs from it
def SampleMI(X, num_pairs, length=200):
    res = []
    for i in range(num_pairs):
        idx = np.random.randint(0, X.shape[0])
        while (idx + length >= X.shape[0]):
            idx = np.random.randint(0, X.shape[0])
        x1 = X[idx:idx + length]
        idx = np.random.randint(0, X.shape[0])
        while (idx + length >= X.shape[0]):
            idx = np.random.randint(0, X.shape[0])
        x2 = X[idx:idx + length]

        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        x1 = x1.view(1, 1, 3, length)
        x2 = x2.view(1, 1, 3, length)
        res.append([x1, x2])
    return res

# generate a batch for phase 2 training
# X[i] is a bag
def take_batch(batch_size, X, Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(0,X.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i:i + batch_size]
        yield X[excerpt], Y[excerpt]


def MILtrain(net, Xtrain, Ytrain, Xval, Yval, batch_size=10, num_pairs=5, epochs=300, lr=0.0005):
    print("training with batchsize = ",str(batch_size), ", num_pairs = ",str(num_pairs))
    loss_func = SiameseNet.bagLoss()
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    best_F1 = 0

    for epoch in range(epochs):
        if (torch.cuda.is_available()):
            #net = nn.DataParallel(net)
            net = net.cuda()
            loss_func = loss_func.cuda()
        train_loss = []
        net.train()
        for x, y in take_batch(batch_size, Xtrain, Ytrain):  # 10x1000x3, 10x1
            labels = torch.from_numpy(y).view(batch_size, -1)
            if (torch.cuda.is_available()):
                labels = labels.cuda()
            y_preds = torch.empty(x.shape[0], num_pairs, 1) # predictions for the whole batch
            opt.zero_grad()
            for i in range(x.shape[0]):
                pairs = SampleMI(x[i], num_pairs)
                y_pred = torch.zeros(num_pairs, 1)
                for j in range(len(pairs)):
                    x1 = pairs[j][0]
                    x2 = pairs[j][1]
                    if (torch.cuda.is_available()):
                        x1 = x1.cuda()
                        x2 = x2.cuda()
                    y_pred[j] = net(x1, x2) # per prediction for each pair from a bag
                y_preds[i, :, :] = y_pred
            loss = loss_func(y_preds, labels.float())
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
        F1, precision, recall, acc = testPerformance(net, Xval, Yval, num_pairs)
        if (F1 >= best_F1):  # store the model performs best on validation set
            best_model = net.state_dict()
            best_F1 = F1
            torch.save(best_model, 'best_MILmodel_2s.pth')
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)),
              "Validation F1: {:.4f}...".format(F1),
              "Precision and Recall on Validation Set: {:.4f}, {:.4f}".format(precision, recall))

        final_model = net.state_dict()
        if ((epoch + 1) % 10 == 0):
            torch.save(final_model, 'final_MILmodel_2s.pth')
    return final_model, best_model


if __name__ == '__main__':

    numpairs = 7
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

    df = open('/home/jxin05/Phase1Data/train.pickle', 'rb')
    X_train = pickle.load(df)
    df.close()

    df = open('/home/jxin05/Phase1Data/train.pickle', 'rb')
    X_train = pickle.load(df)
    df.close()

    df = open('/home/jxin05/Phase1Data/test.pickle', 'rb')
    X_test = pickle.load(df)
    df.close()

    df = open('/home/jxin05/Phase1Data/validation.pickle', 'rb')
    X_val = pickle.load(df)
    df.close()

    net = SiameseNet.SiameseNetwork()
    net.apply(SiameseNet.init_weights)
    # phase 1: train on pairs
    print("Start Phase 1 training...")

    final_state, best_state = trainBatch(net, X_train, X_val, epochs = 160)
    best_model = SiameseNet.SiameseNetwork()
    #best_model = nn.DataParallel(best_model)
    best_model.load_state_dict(best_state)

    print("Finish Phase 1 training!")
    # generate phase 2 training and validation data
    Xtrain, Ytrain, Xval, Yval = DataGeneratePhase2(datapath, trainFiles, valFiles)
    # MIL training
    print("Start Phase 2 training...")

    final_state2, best_state2 = MILtrain(best_model, Xtrain, Ytrain, Xval, Yval, num_pairs=7, epochs=200)

    print("Finish Phase 2 training!")
    best_model2 = SiameseNet.SiameseNetwork()
    best_model2.load_state_dict(best_state2)
    final_model2 = SiameseNet.SiameseNetwork()
    final_model2.load_state_dict(final_state2)
    print("Evaluating on testing group...")
    # generate phase 2 testing data
    Xtest_100 = SyntheticGenerator(datapath, testFiles, 200, 100, 1000)
    Xtest_200 = SyntheticGenerator(datapath, testFiles, 200, 200, 1000)
    Xtest_300 = SyntheticGenerator(datapath, testFiles, 200, 300, 1000)
    Xtest_400 = SyntheticGenerator(datapath, testFiles, 200, 400, 1000)
    Xtest_500 = SyntheticGenerator(datapath, testFiles, 200, 500, 1000)
    XtestPure = ExtractPure(datapath, testFiles, 1000, 200)

    # evaluate both best and final model 
    finalDict = {"acc_Pure": [], "acc_100":[], "acc_200":[], "acc_300": [], "acc_400": [], "acc_500": []}
    bestDict = {"acc_Pure": [], "acc_100": [], "acc_200": [], "acc_300": [], "acc_400": [], "acc_500": []}
    for i in range(10):
        F1, precision, recall, acc = testPerformance(final_model2, XtestPure, np.ones(XtestPure.shape[0]), numpairs)
        finalDict["acc_Pure"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model2, Xtest_100, np.zeros(Xtest_100.shape[0]), numpairs)
        finalDict["acc_100"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model2, Xtest_200, np.zeros(Xtest_200.shape[0]), numpairs)
        finalDict["acc_200"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model2, Xtest_300, np.zeros(Xtest_300.shape[0]), numpairs)
        finalDict["acc_300"].append(acc)
        F1, precision, recall, acc= testPerformance(final_model2, Xtest_400, np.zeros(Xtest_400.shape[0]), numpairs)
        finalDict["acc_400"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model2, Xtest_500, np.zeros(Xtest_500.shape[0]), numpairs)
        finalDict["acc_500"].append(acc)

        F1, precision, recall, acc = testPerformance(best_model2, XtestPure, np.ones(XtestPure.shape[0]), numpairs)
        bestDict["acc_Pure"].append(acc)
        F1, precision, recall, acc = testPerformance(best_model2, Xtest_100, np.zeros(Xtest_100.shape[0]), numpairs)
        bestDict["acc_100"].append(acc)
        F1, precision, recall, acc = testPerformance(best_model2, Xtest_200, np.zeros(Xtest_200.shape[0]), numpairs)
        bestDict["acc_200"].append(acc)
        F1, precision, recall, acc = testPerformance(best_model2, Xtest_300, np.zeros(Xtest_300.shape[0]), numpairs)
        bestDict["acc_300"].append(acc)
        F1, precision, recall, acc = testPerformance(best_model2, Xtest_400, np.zeros(Xtest_400.shape[0]), numpairs)
        bestDict["acc_400"].append(acc)
        F1, precision, recall, acc = testPerformance(best_model2, Xtest_500, np.zeros(Xtest_500.shape[0]), numpairs)
        bestDict["acc_500"].append(acc)

    f = open('final_res.txt', 'w')
    f.write(str(finalDict))
    f.close()

    f = open('best_res.txt', 'w')
    f.write(str(bestDict))
    f.close()












