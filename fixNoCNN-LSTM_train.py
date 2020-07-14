import numpy as np
import torch
from torch import nn
import random
import CNNLSTM

# Train a CNN-LSTM model on both smoothed and non-smoothed data

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

datapath = "/home/jxin05/newCode/Baseline_Dataset/"

def take_batch(batch_size, X, Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(0,X.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i:i + batch_size]
        yield X[excerpt], Y[excerpt]

def predict(net, x):
    with torch.no_grad():
        h = net.init_hidden(1)
        output, h = net(x,h)
    if(output[0,-1,0] >= output[0, -1, 1]):
        return 0
    else:
        return 1

def checkPerformance(net, X, Y):
    if(torch.cuda.is_available()):
        net = net.cuda()
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(X.shape[0]):
        res = predict(net, X[i])
        if(res == Y[i]):
            if(Y[i] == 1):
                TP += 1
            else:
                TN += 1
        else:
            if(Y[i] == 1):
                FN += 1
            else:
                FP += 1
    if (TP == 0):
        precision, recall, F1 = 0, 0, 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
    return F1, precision, recall

def checkAcc(net, X, Y):
    if (torch.cuda.is_available()):
        net = net.cuda()
    correct = 0
    for i in range(X.shape[0]):
        res = predict(net, X[i])
        #print(res, Y[i])
        if(res == Y[i]):
            #print("correct!")
            correct += 1
    return correct/X.shape[0]

def trainBatch(net, trainX, trainY, valX, valY,
               batch_size=40, epochs=300, lr=0.0005):
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    val_F1 = 0
    best_model = None
    for e in range(epochs):
        h = net.init_hidden(batch_size)
        if(torch.cuda.is_available()):
            net = net.cuda()
            criterion = criterion.cuda()
        train_loss = []
        net.train()
        for x, y in take_batch(batch_size, trainX, trainY):
            # print(trainX.shape, trainY.shape)
            targets = torch.from_numpy(y).view(batch_size, -1)
            targets = targets.squeeze()
            h = tuple([each.data for each in h])
            if (torch.cuda.is_available()):
                targets = targets.cuda()
            opt.zero_grad()
            output, h = net(x, h)
            output = output[:, :, -1]
            loss = criterion(output, targets.long())
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
        F1, precision, recall = checkPerformance(net, valX, valY)
        if (F1 >= val_F1):  # store the model performs best on validation set
            best_model = net.state_dict()
            val_F1 = F1
        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)),
              "Validation F1: {:.4f}..., precision: {:.4f}..., recall: {:.4f}...".format(F1, precision, recall))
    final_model = net.state_dict()
    return best_model, final_model

if __name__ == '__main__':

    out = "/home/jxin05/newCode/BaselineRes/nosmooth"

    # trainX = np.load(file = datapath + '/smooth/trainX.npy')
    # trainY = np.load(file = datapath + '/smooth/trainY.npy')
    # valX = np.load(file=datapath + '/smooth/valX.npy')
    # valY = np.load(file=datapath + '/smooth/valY.npy')

    trainX = np.load(file=datapath + '/nosmooth/trainX.npy')
    trainY = np.load(file=datapath + '/nosmooth/trainY.npy')
    valX = np.load(file=datapath + '/nosmooth/valX.npy')
    valY = np.load(file=datapath + '/nosmooth/valY.npy')



    # testX = np.load(file=datapath + '/smooth/testX.npy')
    # testY = np.load(file=datapath + '/smooth/testY.npy')

    testX = np.load(file=datapath + '/nosmooth/testX.npy')
    testY = np.load(file=datapath + '/nosmooth/testY.npy')
    print("num of training samples: {}".format(trainX.shape[0]))
    print("num of testing samples: {}".format(testX.shape[0]))
    print("num of validation samples: {}".format(valX.shape[0]))
	
    net = CNNLSTM.cnn_lstm()
    net.apply(CNNLSTM.init_weights)
    best_state, final_state = trainBatch(net, trainX, trainY, valX, valY, epochs=350)

    best_model = CNNLSTM.cnn_lstm()
    final_model = CNNLSTM.cnn_lstm()

    best_model.load_state_dict(best_state)
    final_model.load_state_dict(final_state)
    bestRes = {"F1": [], "precision": [], "recall": []}

    finalRes = {"F1": [], "precision": [], "recall": []}
	
	# model performance on different types of data
	# 100 means 1 s from a 10 s signal is replaced.
    accSyn = {"100": [], "200": [], "300":[], "400":[], "500":[], "pure": []}
    for i in range(10):
        F1, precision, recall = checkPerformance(best_model, testX, testY)
        bestRes["F1"].append(F1)
        bestRes["precision"].append(precision)
        bestRes["recall"].append(recall)

        F1, precision, recall = checkPerformance(final_model, testX, testY)
        finalRes["F1"].append(F1)
        finalRes["precision"].append(precision)
        finalRes["recall"].append(recall)

        Xpure = testX[0:640, :, :]
        Ypure = np.ones((640, 1))
        accPure = checkAcc(best_model, Xpure, Ypure)
        x100 = testX[640:768, :, :]
        x200 = testX[768:896, :, :]
        x300 = testX[896:1024, :, :]
        x400 = testX[1024:1152, :, :]
        x500 = testX[1152:1280, :, :]
        synY = np.zeros((128, 1))
        acc100 = checkAcc(best_model, x100, synY)
        acc200 = checkAcc(best_model, x200, synY)
        acc300 = checkAcc(best_model, x300, synY)
        acc400 = checkAcc(best_model, x400, synY)
        acc500 = checkAcc(best_model, x500, synY)
        accSyn["100"].append(acc100)
        accSyn["200"].append(acc200)
        accSyn["300"].append(acc300)
        accSyn["400"].append(acc400)
        accSyn["500"].append(acc500)
        accSyn["pure"].append(accPure)

    print("Best model's average F1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(np.mean(bestRes["F1"]),
                                                                                      np.mean(bestRes["precision"]),
                                                                                      np.mean(bestRes["recall"])))
    print("Final model's average F1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(np.mean(finalRes["F1"]),
                                                                                      np.mean(finalRes["precision"]),
                                                                                      np.mean(finalRes["recall"])))
    print("Average accuracy on each synthetic subset - acc100 = {:.4f}, acc200 = {:.4f}, acc300 = {:.4f}, acc400 = {:.4f}, acc500 = {:.4f}".format(np.mean(accSyn["100"]),
                                                                                                                                                   np.mean(accSyn["200"]),
                                                                                                                                                   np.mean(accSyn["300"]),
                                                                                                                                                   np.mean(accSyn["400"]),
                                                                                                                                                   np.mean(accSyn["500"]))
          )
    f = open(out + 'AccOnSyns.txt', 'w')
    f.write(str(accSyn))
    f.close()

    f = open(out + 'best_res.txt', 'w')
    f.write(str(bestRes))
    f.close()

    f = open(out + 'final_res.txt', 'w')
    f.write(str(finalRes))
    f.close()

    out = "/home/jxin05/newCode/BaselineRes/smooth/"

    trainX = np.load(file = datapath + '/smooth/trainX.npy')
    trainY = np.load(file = datapath + '/smooth/trainY.npy')
    valX = np.load(file=datapath + '/smooth/valX.npy')
    valY = np.load(file=datapath + '/smooth/valY.npy')

    # trainX = np.load(file=datapath + '/nosmooth/trainX.npy')
    # trainY = np.load(file=datapath + '/nosmooth/trainY.npy')
    # valX = np.load(file=datapath + '/nosmooth/valX.npy')
    # valY = np.load(file=datapath + '/nosmooth/valY.npy')

    net = CNNLSTM.cnn_lstm()
    net.apply(CNNLSTM.init_weights)
    best_state, final_state = trainBatch(net, trainX, trainY, valX, valY, epochs=350)

    testX = np.load(file=datapath + '/smooth/testX.npy')
    testY = np.load(file=datapath + '/smooth/testY.npy')

    # testX = np.load(file=datapath + '/nosmooth/testX.npy')
    # testY = np.load(file=datapath + '/nosmooth/testY.npy')

    best_model = CNNLSTM.cnn_lstm()
    final_model = CNNLSTM.cnn_lstm()

    best_model.load_state_dict(best_state)
    final_model.load_state_dict(final_state)
    bestRes = {"F1": [], "precision": [], "recall": []}

    finalRes = {"F1": [], "precision": [], "recall": []}

    accSyn = {"100": [], "200": [], "300":[], "400":[], "500":[], "pure": []}
    for i in range(10):
        F1, precision, recall = checkPerformance(best_model, testX, testY)
        bestRes["F1"].append(F1)
        bestRes["precision"].append(precision)
        bestRes["recall"].append(recall)

        F1, precision, recall = checkPerformance(final_model, testX, testY)
        finalRes["F1"].append(F1)
        finalRes["precision"].append(precision)
        finalRes["recall"].append(recall)

        Xpure = testX[0:640, :, :]
        Ypure = np.ones((640,1))
        accPure = checkAcc(best_model, Xpure, Ypure)
        x100 = testX[640:768, :, :]
        x200 = testX[768:896, :, :]
        x300 = testX[896:1024, :, :]
        x400 = testX[1024:1152, :, :]
        x500 = testX[1152:1280, :, :]
        synY = np.zeros((128,1))
        acc100 = checkAcc(best_model, x100, synY)
        acc200 = checkAcc(best_model, x200, synY)
        acc300 = checkAcc(best_model, x300, synY)
        acc400 = checkAcc(best_model, x400, synY)
        acc500 = checkAcc(best_model, x500, synY)
        accSyn["100"].append(acc100)
        accSyn["200"].append(acc200)
        accSyn["300"].append(acc300)
        accSyn["400"].append(acc400)
        accSyn["500"].append(acc500)
        accSyn["pure"].append(accPure)

    print("Best model's average F1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(np.mean(bestRes["F1"]),
                                                                                      np.mean(bestRes["precision"]),
                                                                                      np.mean(bestRes["recall"])))
    print("Final model's average F1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(np.mean(finalRes["F1"]),
                                                                                      np.mean(finalRes["precision"]),
                                                                                      np.mean(finalRes["recall"])))
    print("Average accuracy on each synthetic subset - acc100 = {:.4f}, acc200 = {:.4f}, acc300 = {:.4f}, acc400 = {:.4f}, acc500 = {:.4f}".format(np.mean(accSyn["100"]),
                                                                                                                                                   np.mean(accSyn["200"]),
                                                                                                                                                   np.mean(accSyn["300"]),
                                                                                                                                                   np.mean(accSyn["400"]),
                                                                                                                                                   np.mean(accSyn["500"]))
          )
    f = open(out + 'AccOnSyns.txt', 'w')
    f.write(str(accSyn))
    f.close()

    f = open(out + 'best_res.txt', 'w')
    f.write(str(bestRes))
    f.close()

    f = open(out + 'final_res.txt', 'w')
    f.write(str(finalRes))
    f.close()





