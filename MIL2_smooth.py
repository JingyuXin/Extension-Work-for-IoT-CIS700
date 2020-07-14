import SiameseNet
import random
import numpy as np
import torch
import pandas as pd

#Train a MIL-Siamese model using smoothed synthetic data

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

datapath = "/home/jxin05/rawdata/"

class bag:
    def __init__(self, Xpair_lst, label): # 1x3x200
        self.Xpairs = Xpair_lst
        self.label = label
    def getNumPairs(self):
        return len(self.Xpairs)

def remove_col(df):
    return df.drop(['EID','time','time_in_ms'], axis =1)

def sample(path, file, length, start, end):
    df = pd.read_csv(path + file)
    df = remove_col(df)
    idx = np.random.randint(start, end)
    while(idx + length >= end):
        idx = np.random.randint(start, end)
    res = df.iloc[idx:idx+length].values
    return res

# used for generating data for phase 2 training
# given a list of files in a group
# randomly pick a file and extract a portion with some size
def Extract(usr, path, files, size):
    extractFrom = np.random.randint(0,len(files))
    #print("Extract from user {}".format(files[extractFrom][:-12]))
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
def SyntheticGenerator(path, usr_file, files, num_sample, replace_size, puresize, start, end, ifSmooth = True):
    X = np.zeros((num_sample, puresize, 3))
    usr = usr_file[:-12]
    #print(usr_file)
    for i in range(num_sample):
        puredata = sample(path, usr_file, puresize, start, end)
        copy, smooth_start, smooth_end = CreateSyn(usr, puredata, path, files, replace_size)
        if(ifSmooth):
            copy = smooth(copy, smooth_start)
            copy = smooth(copy, smooth_end)
        X[i,:,:] = copy
    return X

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


# for a session, make 15 label 1 bags and 15 label 0 bags
def makeBags(path, usrSess, files, num = 10, pureSize = 1000, numPairs = 5, ifSmooth = True):
    df = pd.read_csv(path + usrSess)
    df = remove_col(df)

    startpoint = pureSize + 1000 # template, 10s, 15s or 20s. This is 20s template

    first10s = df.iloc[0:startpoint].values

    Xpure = np.zeros((num+int(num/2), pureSize, 3))
    for i in range(num+int(num/2)): # extract pure parts
        x = sample(path, usrSess, pureSize, startpoint, len(df))
        Xpure[i,:,:] = x
    #print("Pure 10s extracted!")
    Xsyn100 = SyntheticGenerator(path, usrSess, files, int(num/5), 100, pureSize, startpoint, len(df), ifSmooth= ifSmooth)
    Xsyn200 = SyntheticGenerator(path, usrSess, files, int(num / 5), 200, pureSize, startpoint, len(df), ifSmooth= ifSmooth)
    Xsyn300 = SyntheticGenerator(path, usrSess, files, int(num / 5), 300, pureSize, startpoint, len(df), ifSmooth= ifSmooth)
    Xsyn400 = SyntheticGenerator(path, usrSess, files, int(num / 5), 400, pureSize, startpoint, len(df), ifSmooth= ifSmooth)
    Xsyn500 = SyntheticGenerator(path, usrSess, files, int(num / 5), 500, pureSize, startpoint, len(df), ifSmooth= ifSmooth)
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

# for each session in a group
# generate 30 bags containing numPairs instances
def makeGroupBag(datapath, groupfiles, numPairs, ifSmooth = True):
    baglst = []
    for file in groupfiles:
        aBaglst = makeBags(datapath, file, groupfiles, numPairs=numPairs, ifSmooth= ifSmooth)
        baglst += aBaglst
    print("Number of bags is {}.".format(len(baglst)))
    return baglst

def testPerformance(groupBag, net):
    num_sample = len(groupBag)
    FN, FP, TP, TN, correct = 0, 0, 0, 0, 0
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
        if(score > 0.5):
            if(aBag.label == 1):
                correct += 1
                TP += 1
            else:
                FP += 1
        else:
            if(aBag.label == 0):
                correct += 1
                TN += 1
            else:
                FN += 1
    if (TP == 0):
        precision, recall, F1 = 0, 0, 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
    acc = correct / num_sample
    return F1, precision, recall, acc

def take_batch(batch_size, groupBag):
    indices = np.arange(len(groupBag))
    np.random.shuffle(indices)
    for i in range(0,len(groupBag)-batch_size+1, batch_size):
        excerpt = indices[i:i + batch_size]
        res = []
        for idx in excerpt:
            res.append(groupBag[idx])
        yield res


def train(net, trainFiles, valFiles, batch_size=10, epochs=300, lr=0.0003, num_pairs = 7, ifSmooth = True):
    loss_func = SiameseNet.bagLoss()
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    best_F1 = 0

    validationBag = makeGroupBag(datapath, valFiles, numPairs=num_pairs,ifSmooth=ifSmooth)

    for epoch in range(epochs):
        trainBag = makeGroupBag(datapath, trainFiles, numPairs=num_pairs, ifSmooth=ifSmooth)
		# training bags are keeping updating becasue sampling is random
		# not quite about whether it is right
        if (torch.cuda.is_available()):
            #net = nn.DataParallel(net)
            net = net.cuda()
            loss_func = loss_func.cuda()
        train_loss = []
        net.train()
        for bags in take_batch(batch_size, trainBag):
            labels = torch.zeros(batch_size, 1)
            for i in range(batch_size):
                if(bags[i].label == 1):
                    labels[i] = 1
            #labels = torch.from_numpy(y).view(batch_size, -1)
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
        F1, precision, recall, acc = testPerformance(validationBag, net)
        if (F1 >= best_F1):  # store the model performs best on validation set
            best_model = net.state_dict()
            best_F1 = F1
            #torch.save(best_model, out + 'best_MILmodel_2s.pth')
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)),
              "Validation F1: {:.4f}...".format(F1),
              "Precision and Recall on Validation Set: {:.4f}, {:.4f}".format(precision, recall))
        final_model = net.state_dict()
        #if ((epoch + 1) % 10 == 0):
            #torch.save(final_model, out + 'final_MILmodel_2s.pth')
    return final_model, best_model

if __name__ == '__main__':
    print("This uses pahse 1 model to do MIL based on that part of user's data (20s) is obtained already. Smoothing used")
    ifSmooth = True
    #modelpath = '/home/jxin05/ComplexNet1/'
    modelpath = '/home/jxin05/newCode/Phase1Model/'
    numPairs = 7
    print("Training with {} pairs".format(numPairs))
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

    #trainBag = makeGroupBag(datapath, trainFiles)
    #validationBag = makeGroupBag(datapath, valFiles)
    #testBag = makeGroupBag(datapath, testFiles)

    print("Loading Phase 1 Model...")
    net = SiameseNet.ComplexSiameseNet()

    net.load_state_dict(torch.load(modelpath + 'phase1_best_model.pth'))
    print("Finish Loading.")

    final_model = SiameseNet.ComplexSiameseNet()
    best_model = SiameseNet.ComplexSiameseNet()

    print("Evaluate before training...")

    for i in range(5):
        testBag = makeGroupBag(datapath, testFiles, numPairs=numPairs, ifSmooth=ifSmooth)
        F1, precision, recall, acc = testPerformance(testBag, net)
        print("Performance on testing set before training - F1:{:.4f}, precision:{:.4f}, recall:{:.4f}".format(F1, precision, recall))

    print("Start Phase 2 training...")

    final_state, best_state = train(net, trainFiles, valFiles, epochs= 150, num_pairs= numPairs, ifSmooth = ifSmooth)
    print("Finish Phase 2 training!")
    #out = "/home/jxin05/MIL-authentication/"


    final_model.load_state_dict(final_state)
    best_model.load_state_dict(best_state)
    print("Evaluating on testing group...")

    for i in range(10):
        testBag = makeGroupBag(datapath, testFiles, numPairs = numPairs, ifSmooth=ifSmooth)
        F1, precision, recall, acc = testPerformance(testBag, final_model)
        print("Final model's performance on testing set after training - F1:{:.4f}, precision:{:.4f}, recall:{:.4f}".format(F1,
                                                                                                               precision,
                                                                                                               recall))

        F1, precision, recall, acc = testPerformance(testBag, best_model)
        print("Best model's performance on testing set after training - F1:{:.4f}, precision:{:.4f}, recall:{:.4f}".format(
                F1,
                precision,
                recall))
    out = "/home/jxin05/newCode/MIL2_authentication/20s/"
    torch.save(final_state, out + 'final_MILmodel.pth')
    torch.save(best_state, out + 'best_MILmodel.pth')






