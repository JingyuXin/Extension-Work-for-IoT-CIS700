import SiameseNet

import random
import numpy as np
import torch
import pandas as pd

# Train a MIL-Siamese model on smoothed data to detect non-pure data

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

def Extract(usr, path, files, size):
    extractFrom = np.random.randint(0,len(files))
    while (files[extractFrom][:-12] == usr):
        extractFrom = np.random.randint(0, len(files))
    df = pd.read_csv(path+files[extractFrom])
    start = np.random.randint(0,len(df)-size)
    x = df.iloc[start:start+size]
    x = remove_col(x)
    return x.values

def ExtractPure(path, files, size, each_num):
    samples = []
    for file in files:
        for i in range(each_num):
            x = sample(path + file, size)
            x = x[np.newaxis, :]  # make it 1 x size x 3
            samples.append(x)
    samples = np.vstack(samples)
    return samples

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

def SyntheticGenerator(path, files, num_each, replace_size, puresize, ifSmooth = True):
    X = []
    for file in files:
        for i in range(num_each):
            usr = file[:-12]
            puredata = sample(path+file, puresize)
            copy, start, end = CreateSyn(usr, puredata, path, files, replace_size)
            if (ifSmooth):
                copy = smooth(copy, start)
                copy = smooth(copy, end)
            copy = copy[np.newaxis, :]
            X.append(copy)
    X = np.vstack(X)
    return X

def DataGeneratePhase2(datapath, trainFiles, valFiles, ifSmooth):
    XtrainPure = ExtractPure(datapath, trainFiles, 1000, 1500)
    Xtrain_100 = SyntheticGenerator(datapath, trainFiles, 300, 100, 1000, ifSmooth= ifSmooth)
    Xtrain_200 = SyntheticGenerator(datapath, trainFiles, 300, 200, 1000, ifSmooth= ifSmooth)
    Xtrain_300 = SyntheticGenerator(datapath, trainFiles, 300, 300, 1000, ifSmooth= ifSmooth)
    Xtrain_400 = SyntheticGenerator(datapath, trainFiles, 300, 400, 1000, ifSmooth= ifSmooth)
    Xtrain_500 = SyntheticGenerator(datapath, trainFiles, 300, 500, 1000, ifSmooth= ifSmooth)

    Xval_100 = SyntheticGenerator(datapath, valFiles, 30, 100, 1000, ifSmooth= ifSmooth)
    Xval_200 = SyntheticGenerator(datapath, valFiles, 30, 200, 1000, ifSmooth= ifSmooth)
    Xval_300 = SyntheticGenerator(datapath, valFiles, 30, 300, 1000, ifSmooth= ifSmooth)
    Xval_400 = SyntheticGenerator(datapath, valFiles, 30, 400, 1000, ifSmooth= ifSmooth)
    Xval_500 = SyntheticGenerator(datapath, valFiles, 30, 500, 1000, ifSmooth= ifSmooth)
    XvalPure = ExtractPure(datapath, valFiles, 1000, 150)

    YvalPure = np.ones(XvalPure.shape[0])
    YvalSyn = np.zeros(Xval_100.shape[0] * 5)
    Yval = np.hstack([YvalPure, YvalSyn])
    Xval = np.vstack([XvalPure, Xval_100, Xval_200, Xval_300, Xval_400, Xval_500])

    YtrainPure = np.ones(XtrainPure.shape[0])
    YtrainSyn = np.zeros(Xtrain_100.shape[0] * 5)
    Ytrain = np.hstack([YtrainPure, YtrainSyn])
    Xtrain = np.vstack([XtrainPure, Xtrain_100, Xtrain_200, Xtrain_300, Xtrain_400, Xtrain_500])

    return Xtrain, Ytrain, Xval, Yval

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
                #print(y_pred)
        score = min(y_pred)
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

# Randomly sample num_pairs pairs from an X
def SampleMI(X, num_pairs, length=200):  # X: 1000x3
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

def take_batch(batch_size, X, Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(0,X.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i:i + batch_size]
        yield X[excerpt], Y[excerpt]

def MILtrain(net, Xtrain, Ytrain, Xval, Yval, out, batch_size=10, num_pairs=5, epochs=300, lr=0.0005):
    print("training with batchsize = ", str(batch_size), ", num_pairs = ", str(num_pairs))
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
            y_preds = torch.empty(x.shape[0], num_pairs, 1)
            #y_preds = torch.empty(x.shape[0], 8, 1)
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
                    y_pred[j] = net(x1, x2)
                y_preds[i, :, :] = y_pred
            loss = loss_func(y_preds, labels.float())
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
        F1, precision, recall, acc = testPerformance(net, Xval, Yval, num_pairs=num_pairs)
        if (F1 >= best_F1):  # store the model performs best on validation set
            best_model = net.state_dict()
            best_F1 = F1
            torch.save(best_model, out + 'Phase2_best_MILmodel_2s.pth')
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)),
              "Validation F1: {:.4f}...".format(F1),
              "Precision and Recall on Validation Set: {:.4f}, {:.4f}".format(precision, recall))

        final_model = net.state_dict()
        if ((epoch + 1) % 10 == 0):
            torch.save(final_model, out + 'Phase2_final_MILmodel_2s.pth')
    return final_model, best_model

if __name__ == '__main__':
    print("This is a testing using balanced dataset with smoothing.")
    modelpath = '/home/jxin05/ComplexNet1/'
    numPairs = 6
    ifSmooth = True
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

    print("Loading Phase 1 Model...")
    net = SiameseNet.ComplexSiameseNet()

    net.load_state_dict(torch.load(modelpath + 'phase1_best_model.pth'))
    # phase 1: train on pairs


    print("Finish Loading.")
    # generate phase 2 data
    # 20 positive sample, 20 negative samples
    print("Generating phase 2 data...")
    num_each = 40


    XtrainPure = ExtractPure(datapath, trainFiles, 1000, int(num_each/2)) # 2800 x 1000 x 3
    Xtrain_100 = SyntheticGenerator(datapath, trainFiles, int(num_each/10), 100, 1000, ifSmooth=ifSmooth) # 560 x 1000 x 3
    Xtrain_200 = SyntheticGenerator(datapath, trainFiles, int(num_each/10), 200, 1000, ifSmooth=ifSmooth)
    Xtrain_300 = SyntheticGenerator(datapath, trainFiles, int(num_each/10), 300, 1000, ifSmooth=ifSmooth)
    Xtrain_400 = SyntheticGenerator(datapath, trainFiles, int(num_each/10), 400, 1000, ifSmooth=ifSmooth)
    Xtrain_500 = SyntheticGenerator(datapath, trainFiles, int(num_each/10), 500, 1000, ifSmooth=ifSmooth)

    Xval_100 = SyntheticGenerator(datapath, valFiles, int(num_each/10), 100, 1000, ifSmooth=ifSmooth) # 80 x 1000 x 3
    Xval_200 = SyntheticGenerator(datapath, valFiles, int(num_each/10), 200, 1000, ifSmooth=ifSmooth)
    Xval_300 = SyntheticGenerator(datapath, valFiles, int(num_each/10), 300, 1000, ifSmooth=ifSmooth)
    Xval_400 = SyntheticGenerator(datapath, valFiles, int(num_each/10), 400, 1000, ifSmooth=ifSmooth)
    Xval_500 = SyntheticGenerator(datapath, valFiles, int(num_each/10), 500, 1000, ifSmooth=ifSmooth)
    XvalPure = ExtractPure(datapath, valFiles, 1000, int(num_each/2)) # 400 x 1000 x 3

    YvalPure = np.ones(XvalPure.shape[0])
    YvalSyn = np.zeros(Xval_100.shape[0] * 5)
    Yval = np.hstack([YvalPure, YvalSyn])
    Xval = np.vstack([XvalPure, Xval_100, Xval_200, Xval_300, Xval_400, Xval_500])

    YtrainPure = np.ones(XtrainPure.shape[0])
    YtrainSyn = np.zeros(Xtrain_100.shape[0] * 5)
    Ytrain = np.hstack([YtrainPure, YtrainSyn])
    Xtrain = np.vstack([XtrainPure, Xtrain_100, Xtrain_200, Xtrain_300, Xtrain_400, Xtrain_500])

    Xtest_100 = SyntheticGenerator(datapath, testFiles, int(num_each/10), 100, 1000, ifSmooth=ifSmooth) # 128 x 1000 x 3
    Xtest_200 = SyntheticGenerator(datapath, testFiles, int(num_each/10), 200, 1000, ifSmooth=ifSmooth)
    Xtest_300 = SyntheticGenerator(datapath, testFiles, int(num_each/10), 300, 1000, ifSmooth=ifSmooth)
    Xtest_400 = SyntheticGenerator(datapath, testFiles, int(num_each/10), 400, 1000, ifSmooth=ifSmooth)
    Xtest_500 = SyntheticGenerator(datapath, testFiles, int(num_each/10), 500, 1000, ifSmooth=ifSmooth)
    XtestPure = ExtractPure(datapath, testFiles, 1000, int(num_each/2)) # 640 x 1000 x 3

    Xtest = np.vstack([XtestPure, Xtest_100, Xtest_200, Xtest_300, Xtest_400, Xtest_500])
    YtestPure = np.ones(XtestPure.shape[0])
    YtestSyn = np.zeros(Xtest_100.shape[0] * 5)
    Ytest = np.hstack([YtestPure, YtestSyn])

    print("Data generated.")

    final_model = SiameseNet.ComplexSiameseNet()
    best_model = SiameseNet.ComplexSiameseNet()
    print("Evaluate before training...")


    for i in range(1):
        F1, precision, recall, acc = testPerformance(net, XtestPure, np.ones(XtestPure.shape[0]), numPairs)
        print("Acc on pure test set:", str(acc))
        F1, precision, recall, acc = testPerformance(net, Xtest_100, np.zeros(Xtest_100.shape[0]), numPairs)
        print("Acc on Syn_100 test set:", str(acc))
        F1, precision, recall, acc = testPerformance(net, Xtest_200, np.zeros(Xtest_200.shape[0]), numPairs)
        print("Acc on Syn_200 test set:", str(acc))
        F1, precision, recall, acc = testPerformance(net, Xtest_300, np.zeros(Xtest_300.shape[0]), numPairs)
        print("Acc on Syn_300 test set:", str(acc))
        F1, precision, recall, acc = testPerformance(net, Xtest_400, np.zeros(Xtest_400.shape[0]), numPairs)
        print("Acc on Syn_400 test set:", str(acc))
        F1, precision, recall, acc = testPerformance(net, Xtest_500, np.zeros(Xtest_500.shape[0]), numPairs)
        print("Acc on Syn_500 test set:", str(acc))
        F1, precision, recall, acc = testPerformance(net, Xtest, Ytest, numPairs)
        print("F1: {}, precision: {}, recall: {}".format(F1, precision, recall))
    # MIL training
    print("Start Phase 2 training...")

    out = '/home/jxin05/newCode/MIL1/smooth/6pairs/'
    final_state, best_state = MILtrain(net, Xtrain, Ytrain, Xval, Yval, batch_size=20, out = out,
                                       num_pairs=numPairs, epochs=200)


    print("Finish Phase 2 training!")


    final_model.load_state_dict(final_state)
    best_model.load_state_dict(best_state)
    print("Evaluating on testing group...")



    finalDict = {"acc_Pure": [], "acc_100":[], "acc_200":[], "acc_300": [], "acc_400": [], "acc_500": [],
                 "F1":[], "precision": [], "recall": []}
    bestDict = {"acc_Pure": [], "acc_100": [], "acc_200": [], "acc_300": [], "acc_400": [], "acc_500": [],
                "F1":[], "precision": [], "recall": []}
    for i in range(10):
        F1, precision, recall, acc = testPerformance(final_model, XtestPure, np.ones(XtestPure.shape[0]),numPairs)
        finalDict["acc_Pure"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model, Xtest_100, np.zeros(Xtest_100.shape[0]), numPairs)
        finalDict["acc_100"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model, Xtest_200, np.zeros(Xtest_200.shape[0]), numPairs)
        finalDict["acc_200"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model, Xtest_300, np.zeros(Xtest_300.shape[0]), numPairs)
        finalDict["acc_300"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model, Xtest_400, np.zeros(Xtest_400.shape[0]), numPairs)
        finalDict["acc_400"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model, Xtest_500, np.zeros(Xtest_500.shape[0]), numPairs)
        finalDict["acc_500"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model, Xtest, Ytest, numPairs)
        print("Final - F1: {}, precision: {}, recall: {}".format(F1, precision, recall))
        finalDict["F1"].append(F1)
        finalDict["precision"].append(precision)
        finalDict["recall"].append(recall)

        F1, precision, recall, acc = testPerformance(best_model, XtestPure, np.ones(XtestPure.shape[0]),numPairs )
        bestDict["acc_Pure"].append(acc)
        F1, precision, recall, acc = testPerformance(best_model, Xtest_100, np.zeros(Xtest_100.shape[0]), numPairs)
        bestDict["acc_100"].append(acc)
        F1, precision, recall, acc = testPerformance(best_model, Xtest_200, np.zeros(Xtest_200.shape[0]), numPairs)
        bestDict["acc_200"].append(acc)
        F1, precision, recall, acc = testPerformance(best_model, Xtest_300, np.zeros(Xtest_300.shape[0]),numPairs)
        bestDict["acc_300"].append(acc)
        F1, precision, recall, acc = testPerformance(best_model, Xtest_400, np.zeros(Xtest_400.shape[0]),numPairs)
        bestDict["acc_400"].append(acc)
        F1, precision, recall, acc = testPerformance(best_model, Xtest_500, np.zeros(Xtest_500.shape[0]),numPairs)
        bestDict["acc_500"].append(acc)
        F1, precision, recall, acc = testPerformance(final_model, Xtest, Ytest, numPairs)
        print("Best - F1: {}, precision: {}, recall: {}".format(F1, precision, recall))
        bestDict["F1"].append(F1)
        bestDict["precision"].append(precision)
        bestDict["recall"].append(recall)
    f = open(out + 'Phase2_final_res.txt', 'w')
    f.write(str(finalDict))
    f.close()

    f = open(out + 'Phase2_best_res.txt', 'w')
    f.write(str(bestDict))
    f.close()