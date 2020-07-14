import SiameseNet
import random
import numpy as np
import torch
import pandas as pd

# Personalization for MIL-Siamese model to do non-pure data detection

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remove_col(df):
    return df.drop(['EID', 'time', 'time_in_ms'], axis=1)

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

def SyntheticGenerator(path, usr_file, files, num_sample, replace_size, puresize, ifSmooth = True):
    X = np.zeros((num_sample, puresize, 3))
    for i in range(num_sample):
        puredata = sample(path+usr_file, puresize)
        usr = usr_file[:-12]
        copy, start, end = CreateSyn(usr, puredata, path, files, replace_size)
        if(ifSmooth):
            copy = smooth(copy, start)
            copy = smooth(copy, end)
        X[i,:,:] = copy
    return X


def PesonalDataset(path, usrfile, files, numPure, numSyn, puresize, ifSmooth):
    Xpure = np.zeros((numPure, puresize, 3))
    for i in range(numPure):
        x = sample(path+usrfile, puresize)
        Xpure[i, :, :] = x
    Xsyn100 = SyntheticGenerator(path, usrfile, files, int(numSyn/5), 100, puresize, ifSmooth= ifSmooth)
    Xsyn200 = SyntheticGenerator(path, usrfile, files, int(numSyn/5), 200, puresize, ifSmooth= ifSmooth)
    Xsyn300 = SyntheticGenerator(path, usrfile, files, int(numSyn/5), 300, puresize, ifSmooth= ifSmooth)
    Xsyn400 = SyntheticGenerator(path, usrfile, files, int(numSyn/5), 400, puresize, ifSmooth= ifSmooth)
    Xsyn500 = SyntheticGenerator(path, usrfile, files, int(numSyn/5), 500, puresize, ifSmooth= ifSmooth)

    Ypure = np.ones(Xpure.shape[0])
    Ysyn = np.zeros(Xsyn100.shape[0] * 5)
    Y = np.hstack([Ypure, Ysyn])
    X = np.vstack([Xpure, Xsyn100, Xsyn200, Xsyn300, Xsyn400, Xsyn500])
    return X, Y


def take_batch(batch_size, X, Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[i:i + batch_size]
        yield X[excerpt], Y[excerpt]


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


def testPerformance(net, X, Y, num_pairs):
    num_sample = X.shape[0]
    correct = 0
    FN = 0
    FP = 0
    TP = 0
    TN = 0
    num_pos = 0
    num_neg = 0
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
                # print(y_pred)
        score = min(y_pred)
        # print(score)
        if (score > 0.5):
            if (Y[i] == 1):
                correct += 1
                TP += 1
                num_pos += 1
            else:
                num_neg += 1
                FP += 1
        else:
            if (Y[i] == 0):
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


def train(net, usr, trainX, trainY, testX, testY, outdir, after_dict,
          batch_size=1, num_pairs=7, epochs=20, lr=0.0002):
    print("personalized training on user ", usr,
          "with batchsize = ", str(batch_size), ", num_pairs = ", str(num_pairs))
    loss_func = SiameseNet.bagLoss()
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        if (torch.cuda.is_available()):
            net = net.cuda()
            loss_func = loss_func.cuda()
        train_loss = []
        net.train()
        for x, y in take_batch(batch_size, trainX, trainY):  # 10x1000x3, 10x1
            labels = torch.from_numpy(y).view(batch_size, -1)
            if (torch.cuda.is_available()):
                labels = labels.cuda()
            y_preds = torch.empty(x.shape[0], num_pairs, 1)
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

        print("Epoch: {}/{}...".format(epoch, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)))

        if ((epoch) % 1 == 0):
            after_dict[usr][epoch] = {}
            print("evaluate after training {} epochs".format(epoch))
            F1_lst, precision_lst, recall_lst, acc_lst = [], [], [], []
            for j in range(10):
                F1, precision, recall, acc = testPerformance(net, testX, testY, numpairs)
                F1_lst.append(F1)
                precision_lst.append(precision)
                recall_lst.append(recall)
                acc_lst.append(acc)
            after_dict[usr][epoch]['F1'] = np.mean(F1_lst)
            after_dict[usr][epoch]['precision'] = np.mean(precision_lst)
            after_dict[usr][epoch]['recall'] = np.mean(recall_lst)
            after_dict[usr][epoch]['acc'] = np.mean(acc_lst)
    final_state = net.state_dict()
    return after_dict


if __name__ == '__main__':
    outdir = '/home/jxin05/newCode/MIL1_personalization/'
    modelpath = '/home/jxin05/newCode/MIL1/smooth/'
    datapath = "/home/jxin05/rawdata/"
    ifSmooth = True
    numpairs = 7
    print("This experiment is for personalization using MIL1, ifSmooth = {}, numpairs = {}. Train on session 2 and test on session 1".format(ifSmooth, numpairs))
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

    personalization_lst = valFiles + testFiles

    before = {}
    after = {}
    for i in range(len(testFiles)):
        usr = testFiles[i][:-12]
		# personalize the model for each user in test group
        if usr in before.keys():
            continue
        # load best model after phase 2
        net = SiameseNet.ComplexSiameseNet()
        net.apply(SiameseNet.init_weights)
        best_state = torch.load(modelpath + 'Phase2_best_MILmodel_2s.pth')
        net.load_state_dict(best_state)

        session1 = usr + 'session1.csv'
        session2 = usr + 'session2.csv'
        print("Personalize on user {}'s data".format(usr))
        before[usr] = {}
        after[usr] = {}
        trainX, trainY = PesonalDataset(datapath, session2, trainFiles, 30, 30, 1000, ifSmooth= ifSmooth)
		# generate train data based on session 2 and training group
        testX, testY = PesonalDataset(datapath, session1, testFiles, 30, 30, 1000, ifSmooth = ifSmooth)
		# generate test data based on session 1 and testing group
        print("Evaluate before training...")
        F1_lst, precision_lst, recall_lst, acc_lst = [], [], [], []
        for j in range(10):
            F1, precision, recall, acc = testPerformance(net, testX, testY, numpairs)
            F1_lst.append(F1)
            precision_lst.append(precision)
            recall_lst.append(recall)
            acc_lst.append(acc)
        before[usr]['F1'] = np.mean(F1_lst)
        before[usr]['precision'] = np.mean(precision_lst)
        before[usr]['recall'] = np.mean(recall_lst)
        before[usr]['acc'] = np.mean(acc_lst)

        file = open(outdir + 'before_res.txt', 'w')
        file.write(str(before))
        file.close()

        after = train(net, usr, trainX, trainY, testX, testY, outdir, after)

        file = open(outdir + 'after_res.txt', 'w')
        file.write(str(after))
        file.close()

        print("Result for user {} saved.".format(usr))
        # final_state = train(net, usr, trainX, trainY, outdir)
        # net.load_state_dict(final_state)
        # print("evaluate after training...")
        # F1_lst, precision_lst, recall_lst, acc_lst = [], [], [], []
        # for j in range(10):
        #     F1, precision, recall, acc = testPerformance(net, testX, testY, numpairs)
        #     F1_lst.append(F1)
        #     precision_lst.append(precision)
        #     recall_lst.append(recall)
        #     acc_lst.append(acc)
        # after[usr]['F1'] = np.mean(F1_lst)
        # after[usr]['precision'] = np.mean(precision_lst
        # after[usr]['recall'] = np.mean(recall_lst)
        # after[usr]['acc'] = np.mean(acc_lst)

        # file = open(outdir + 'after_res.txt', 'w')
        # file.write(str(after))
        # file.close()







