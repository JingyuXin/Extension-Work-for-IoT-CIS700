
import AttentionSiameseMIL
import random
import numpy as np
import torch
import pandas as pd

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

def testPerformance(net, X, Y, num_pairs):
    if (torch.cuda.is_available()):
        net = net.cuda()
    net.eval()
    num_sample = X.shape[0]
    correct = 0
    FN = 0
    FP = 0
    TP = 0
    TN = 0
    for i in range(num_sample):
        pairs = SampleMI(X[i], num_pairs)
        N = len(pairs)
        _, _, h, w = pairs[0][0].shape
        x1s = torch.zeros(N, 1, h, w)
        x2s = torch.zeros(N, 1, h, w)
        for j in range(N):
            x1 = pairs[j][0]
            x2 = pairs[j][1]
            x1s[j, 0, :, :] = x1
            x2s[j, 0, :, :] = x2

        if (torch.cuda.is_available()):
            x1s = x1s.cuda()
            x2s = x2s.cuda()
        Y_prob, Y_hat, A = net(x1s, x2s)
        score = Y_hat
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

def SampleMI2(X, overlap = 0.25, length=200):
    x_lst = []
    start = 0
    size = X.shape[0]
    res = []
    while (start + length <= size):
        x = X[start:start + length, :]
        start += int((1-overlap) * length)
        x_lst.append(x)
    for i in range(len(x_lst)):
        for j in range(i+1, len(x_lst)):
            x1 = x_lst[i]
            x2 = x_lst[j]
            x1 = torch.from_numpy(x1).float()
            x2 = torch.from_numpy(x2).float()
            x1 = x1.view(1, 1, 3, length)
            x2 = x2.view(1, 1, 3, length)
            res.append([x1, x2])
    assert (len(res) == 15)
    return res

def take_batch(batch_size, X, Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(0,X.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i:i + batch_size]
        yield X[excerpt], Y[excerpt]


def MILtrain(net, Xtrain, Ytrain, Xval, Yval, out, batch_size=10, num_pairs=5, epochs=300, lr=0.0005):
    print("training with batchsize = ", str(batch_size), ", num_pairs = ", str(num_pairs))
    loss_func = torch.nn.BCELoss()
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.9)
    best_F1 = 0

    for epoch in range(epochs):
        if (torch.cuda.is_available()):
            net = net.cuda()
            loss_func = loss_func.cuda()
        train_loss = []
        net.train()
        for x, y in take_batch(batch_size, Xtrain, Ytrain):
            labels = torch.from_numpy(y).view(batch_size, -1)
            if (torch.cuda.is_available()):
                labels = labels.cuda()
            y_preds = torch.empty(x.shape[0], 1)
            # y_preds = torch.empty(x.shape[0], 8, 1)
            opt.zero_grad()
            for i in range(x.shape[0]):
                pairs = SampleMI(x[i], num_pairs)
                N = len(pairs)
                _, _, h, w = pairs[0][0].shape
                x1s = torch.zeros(N, 1, h, w)
                x2s = torch.zeros(N, 1, h, w)
                for j in range(N):
                    x1 = pairs[j][0]
                    x2 = pairs[j][1]
                    x1s[j, 0, :, :] = x1
                    x2s[j, 0, :, :] = x2

                if (torch.cuda.is_available()):
                    x1s = x1s.cuda()
                    x2s = x2s.cuda()
                Y_prob, Y_hat, A = net(x1s, x2s)
                y_preds[i, :] = Y_prob
            if (torch.cuda.is_available()):
                y_preds = y_preds.cuda()
            loss = loss_func(y_preds.double(), labels.double())
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
        scheduler.step()
        F1, precision, recall, acc = testPerformance(net, Xval, Yval, num_pairs=num_pairs)
        if (F1 >= best_F1):  # store the model performs best on validation set
            best_model = net.state_dict()
            best_F1 = F1
            torch.save(best_model, out + 'Att' + str(num_pairs) + 'pair_Best_MILmodel.pth')

        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)),
              "Validation F1: {:.4f}...".format(F1),
              "Precision and Recall on Validation Set: {:.4f}, {:.4f}".format(precision, recall))
        final_model = net.state_dict()
        if ((epoch + 1) % 10 == 0):
            torch.save(final_model, out + 'Att-'+ str(num_pairs) + 'pair_Final_MILmodel.pth')
    return final_model, best_model

if __name__ == '__main__':
    numPairs = 15
    ifSmooth = True
    print("Number of pairs in training: {}, ifSmooth = {}".format(numPairs, ifSmooth))
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
    net = AttentionSiameseMIL.Attention()
    #state = torch.load('/home/jxin05/newCode/MIL1/nosmooth/7pairs/7pairs_best_MILmodel_2s.pth')
    #net.load_state_dict(state)
    #net.apply(AttentionSiameseMIL.init_weights)
    #net.load_state_dict(torch.load(modelpath + 'phase1_best_model.pth'))

    print("Finish Loading.")
    # Load phase 2 data
    print("Loading data...")
    if(ifSmooth):
        datapath = '/home/jxin05/BBMAS-split/forgeReplay2/smooth/'
        outpath = '/home/jxin05/BBMAS-split/forgeReplay2Res/' + str(numPairs) + 'pair/smooth/'
    else:
        datapath = '/home/jxin05/BBMAS-split/forgeReplay2/nosmooth/'
        outpath = '/home/jxin05/BBMAS-split/forgeReplay2Res/' + str(numPairs) + 'pair/nosmooth/'


    XtrainPure = np.load(datapath + 'trainPos.npy')
    Xtrain_100 = np.load(datapath + 'trainAtt100.npy')
    Xtrain_200 = np.load(datapath + 'trainAtt200.npy')
    Xtrain_300 = np.load(datapath + 'trainAtt300.npy')
    Xtrain_400 = np.load(datapath + 'trainAtt400.npy')
    Xtrain_500 = np.load(datapath + 'trainAtt500.npy')

    YtrainPure = np.ones(XtrainPure.shape[0])
    YtrainSyn = np.zeros(Xtrain_100.shape[0] * 5)
    Ytrain = np.hstack([YtrainPure, YtrainSyn])
    Xtrain = np.vstack([XtrainPure, Xtrain_100, Xtrain_200, Xtrain_300, Xtrain_400, Xtrain_500])

    XvalPure = np.load(datapath + 'valPos.npy')
    Xval_100 = np.load(datapath + 'valAtt100.npy')
    Xval_200 = np.load(datapath + 'valAtt200.npy')
    Xval_300 = np.load(datapath + 'valAtt300.npy')
    Xval_400 = np.load(datapath + 'valAtt400.npy')
    Xval_500 = np.load(datapath + 'valAtt500.npy')

    YvalPure = np.ones(XvalPure.shape[0])
    YvalSyn = np.zeros(Xval_100.shape[0] * 5)
    Yval = np.hstack([YvalPure, YvalSyn])
    Xval = np.vstack([XvalPure, Xval_100, Xval_200, Xval_300, Xval_400, Xval_500])

    XtestPure = np.load(datapath + 'testPos.npy')
    Xtest_100 = np.load(datapath + 'testAtt100.npy')
    Xtest_200 = np.load(datapath + 'testAtt200.npy')
    Xtest_300 = np.load(datapath + 'testAtt300.npy')
    Xtest_400 = np.load(datapath + 'testAtt400.npy')
    Xtest_500 = np.load(datapath + 'testAtt500.npy')

    Xtest = np.vstack([XtestPure, Xtest_100, Xtest_200, Xtest_300, Xtest_400, Xtest_500])
    YtestPure = np.ones(XtestPure.shape[0])
    YtestSyn = np.zeros(Xtest_100.shape[0] * 5)
    Ytest = np.hstack([YtestPure, YtestSyn])

    print("Data Loaded.")

    best_model = AttentionSiameseMIL.Attention()
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

    final_state, best_state = MILtrain(net, Xtrain, Ytrain, Xval, Yval,batch_size=20, out = outpath,
                                       num_pairs=numPairs, epochs=150, lr = 0.0005)


    print("Finish Phase 2 training!")


    best_model.load_state_dict(best_state)
    print("Evaluating on testing group...")


    bestDict = {"acc_Pure": [], "acc_100": [], "acc_200": [], "acc_300": [], "acc_400": [], "acc_500": [],
                "F1":[], "precision": [], "recall": []}
    for i in range(3):

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
        F1, precision, recall, acc = testPerformance(best_model, Xtest, Ytest, numPairs)
        print("Best - F1: {}, precision: {}, recall: {}".format(F1, precision, recall))
        bestDict["F1"].append(F1)
        bestDict["precision"].append(precision)
        bestDict["recall"].append(recall)
    print("Average F1 : {}, precision: {}, recall: {}".format(np.mean(bestDict["F1"]), np.mean(bestDict["precision"]),
                                                              np.mean(bestDict["recall"])))

    f = open(outpath + 'Att-best_res.txt', 'w')
    f.write(str(bestDict))
    f.close()

    for key in bestDict.keys():
        print("Average " + key + ": {}".format(np.mean(bestDict[key])))