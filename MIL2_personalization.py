import SiameseNet
import random
import numpy as np
import torch
import pandas as pd

# Personalization for MIL-Siamese model to do user authentication

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
    #print(usr_file)
    for i in range(num_sample):
        puredata = sample(path, usr_file, puresize, start)
        copy, smooth_start, smooth_end = CreateSyn(usr, puredata, path, files, replace_size)
        if(ifSmooth):
            copy = smooth(copy, smooth_start)
            copy = smooth(copy, smooth_end)
        X[i,:,:] = copy
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

# for a session, make 10 label 1 bags and 10 label 0 bags
def makeBags(path, usrSess, files, num = 10, pureSize = 1000, numPairs = 5, ifSmooth = True):
    df = pd.read_csv(path + usrSess)
    df = remove_col(df)

    startpoint = pureSize + 1000

    first10s = df.iloc[0:startpoint].values

    Xpure = np.zeros((num+int(num/2), pureSize, 3))
    for i in range(num+int(num/2)): # extract pure parts
        x = sample(path, usrSess, pureSize, startpoint)
        Xpure[i,:,:] = x
    #print("Pure 10s extracted!")
    Xsyn100 = SyntheticGenerator(path, usrSess, files, int(num/5), 100, pureSize, startpoint,  ifSmooth= ifSmooth)
    Xsyn200 = SyntheticGenerator(path, usrSess, files, int(num / 5), 200, pureSize, startpoint,  ifSmooth= ifSmooth)
    Xsyn300 = SyntheticGenerator(path, usrSess, files, int(num / 5), 300, pureSize, startpoint,  ifSmooth= ifSmooth)
    Xsyn400 = SyntheticGenerator(path, usrSess, files, int(num / 5), 400, pureSize, startpoint,  ifSmooth= ifSmooth)
    Xsyn500 = SyntheticGenerator(path, usrSess, files, int(num / 5), 500, pureSize, startpoint,  ifSmooth= ifSmooth)
    Xsyn = np.vstack([Xsyn100, Xsyn200, Xsyn300, Xsyn400, Xsyn500])
    #print("Synthetic 10s generated!")

    imposter = np.zeros((int(num/2), pureSize, 3))
    usr = usrSess[:-12]
    for i in range(int(num/2)):
        imposter[i, :, :] = Extract(usr, path, files, pureSize)
    bagLst = []
    for i in range(int(1.5*num)):
        #print("Createing bag No. {}".format(i))
        pairs = makePairs(first10s, Xpure[i, :, :], numPairs)
        posBag = bag(pairs, 1)
        #print("number of pairs in a posBag is {}".format(posBag.getNumPairs()))
        bagLst.append(posBag)
    for i in range(num):
        pairs = makePairs(first10s, Xsyn[i, :, :], numPairs)
        negBag = bag(pairs, 0)
        #print("number of pairs in a negBag is {}".format(negBag.getNumPairs()))
        bagLst.append(negBag)
    for i in range(int(num/2)):
        pairs = makePairs(first10s, imposter[i, :, :], numPairs)
        negBag = bag(pairs, 0)
        bagLst.append(negBag)
    assert(len(bagLst) == 30)
    return bagLst

def makeTestBags(path, template, usrTestSess, files, num = 10, pureSize = 1000, numPairs = 7, ifSmooth = True):
    print("Length of template is {}".format(len(template)))
    Xpure = np.zeros((num + int(num / 2), pureSize, 3))
    for i in range(num + int(num / 2)):  # extract pure parts
        x = sample(path, usrTestSess, pureSize, 0)
        Xpure[i, :, :] = x
    Xsyn100 = SyntheticGenerator(path, usrTestSess, files, int(num / 5), 100, pureSize, 0, ifSmooth=ifSmooth)
    Xsyn200 = SyntheticGenerator(path, usrTestSess, files, int(num / 5), 200, pureSize, 0, ifSmooth=ifSmooth)
    Xsyn300 = SyntheticGenerator(path, usrTestSess, files, int(num / 5), 300, pureSize, 0, ifSmooth=ifSmooth)
    Xsyn400 = SyntheticGenerator(path, usrTestSess, files, int(num / 5), 400, pureSize, 0, ifSmooth=ifSmooth)
    Xsyn500 = SyntheticGenerator(path, usrTestSess, files, int(num / 5), 500, pureSize, 0, ifSmooth=ifSmooth)
    Xsyn = np.vstack([Xsyn100, Xsyn200, Xsyn300, Xsyn400, Xsyn500])

    imposter = np.zeros((int(num / 2), pureSize, 3))
    usr = usrTestSess[:-12]
    for i in range(int(num / 2)):
        imposter[i, :, :] = Extract(usr, path, files, pureSize)
    bagLst = []
    for i in range(int(1.5 * num)):
        # print("Createing bag No. {}".format(i))
        pairs = makePairs(template, Xpure[i, :, :], numPairs)
        posBag = bag(pairs, 1)
        # print("number of pairs in a posBag is {}".format(posBag.getNumPairs()))
        bagLst.append(posBag)
    for i in range(num):
        pairs = makePairs(template, Xsyn[i, :, :], numPairs)
        negBag = bag(pairs, 0)
        # print("number of pairs in a negBag is {}".format(negBag.getNumPairs()))
        bagLst.append(negBag)
    for i in range(int(num / 2)):
        pairs = makePairs(template, imposter[i, :, :], numPairs)
        negBag = bag(pairs, 0)
        bagLst.append(negBag)
    assert (len(bagLst) == 30)
    return bagLst


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


def train(net, path, train_usrfile, test_usrfile, trainFiles, testFiles, outdir, after_dict,
          batch_size=1, num_pairs=7, epochs=20, lr=0.0002):
    print("personalized training on user ", usr,
          "with batchsize = ", str(batch_size), ", num_pairs = ", str(num_pairs))
    loss_func = SiameseNet.bagLoss()
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)

    df = pd.read_csv(path + train_usrfile)
    df = remove_col(df)

    startpoint =  2000

    template = df.iloc[0:startpoint].values

    for epoch in range(1,epochs+1):
        trainBag = makeBags(path, train_usrfile, trainFiles, numPairs=num_pairs, ifSmooth=ifSmooth)

        if (torch.cuda.is_available()):
            # net = nn.DataParallel(net)
            net = net.cuda()
            loss_func = loss_func.cuda()
        train_loss = []
        net.train()
        for bags in take_batch(batch_size, trainBag):
            labels = torch.zeros(batch_size, 1)
            for i in range(batch_size):
                if (bags[i].label == 1):
                    labels[i] = 1
            # labels = torch.from_numpy(y).view(batch_size, -1)
            if (torch.cuda.is_available()):
                labels = labels.cuda()
            y_preds = torch.empty(batch_size, num_pairs, 1)
            opt.zero_grad()
            for j in range(batch_size):
                y_pred = torch.zeros(num_pairs, 1)
                for k in range(num_pairs):
                    x1 = bags[j].Xpairs[k][0]
                    x2 = bags[j].Xpairs[k][1]
                    if (torch.cuda.is_available()):
                        x1 = x1.cuda()
                        x2 = x2.cuda()
                    y_pred[k] = net(x1, x2)
                y_preds[j, :, :] = y_pred
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
                testBag = makeBags(datapath, testSession, testFiles, numPairs=numpairs, ifSmooth=ifSmooth)
                #testBag = makeTestBags(path, template, test_usrfile, testFiles, 10, 1000, numPairs=numpairs, ifSmooth=ifSmooth)
                F1, precision, recall, acc = testPerformance(testBag, net)
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
    outdir = '/home/jxin05/newCode/MIL2_personalization/'
    modelpath = "/home/jxin05/newCode/MIL2_authentication/20s/"
    datapath = "/home/jxin05/rawdata/"
    ifSmooth = True
    numpairs = 7
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
        best_state = torch.load(modelpath + 'best_MILmodel.pth')
        net.load_state_dict(best_state)

        session1 = usr + 'session1.csv'
        session2 = usr + 'session2.csv'
        print("Personalize on user {}'s data".format(usr))
        before[usr] = {}
        after[usr] = {}

        trainSession = session1
        testSession = session2
        df = pd.read_csv(datapath + trainSession)
        df = remove_col(df)

        startpoint = 2000

        template = df.iloc[0:startpoint].values

        print("Evaluate before training...")
        F1_lst, precision_lst, recall_lst, acc_lst = [], [], [], []
        for j in range(10):
            testBag = makeBags(datapath, testSession, testFiles, numPairs=numpairs, ifSmooth = ifSmooth)

            F1, precision, recall, acc = testPerformance(testBag, net)
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

        after = train(net, datapath, trainSession, testSession, trainFiles, testFiles, outdir, after)

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







