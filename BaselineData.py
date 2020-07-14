import pandas as pd
import os
import time
import fnmatch
import numpy as np
import torch
import random

# This file is used to generate 10 s pure or non-pure signal for CNN-LSTM model
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

datapath = "/home/jxin05/rawdata/"

# remove some columns from raw .csv file
def remove_col(df):
    return df.drop(['EID','time','time_in_ms'], axis =1)

# used to sample pure samples
def sample(path, file, length):
    df = pd.read_csv(path+file)
    df = remove_col(df)
    idx = np.random.randint(0, len(df))
    while(idx + length >= len(df)):
        idx = np.random.randint(0, len(df))
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

def ExtractPure(path, files, size, num_each):
    samples = []
    for file in files:
        for i in range(num_each):
            x = sample(path, file, size) # size x 3
            x = x[np.newaxis,:] # make it 1 x size x 3
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

def SyntheticGenerator(path, files, replace_size, puresize, num_each, ifSmooth):
    X = []
    for file in files:
        for i in range(num_each):
            usr = file[:-12]
            puredata = sample(path, file, puresize)
            copy, start, end = CreateSyn(usr, puredata, path, files, replace_size)
            if (ifSmooth):
                copy = smooth(copy, start)
                copy = smooth(copy, end)
            copy = copy[np.newaxis, :]
            X.append(copy)
    X = np.vstack(X)
    return X

if __name__ == '__main__':
    out = "/home/jxin05/newCode/Baseline_Dataset/"

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

    print(len(trainFiles), len(testFiles), len(valFiles))
    # pure has label 1, synthetic has label 0
    # No of samples for each file
    # 20 positive sample, 20 negative samples
    num_each = 40

    PureTrainSamples = ExtractPure(datapath, trainFiles, 1000, int(num_each/2)) # 2800 x 1000 x 3
    PureTestSamples = ExtractPure(datapath, testFiles, 1000, int(num_each/2)) # 400 x 1000 x 3
    PureValSamples = ExtractPure(datapath, valFiles, 1000, int(num_each / 2)) # 640 x 1000 x 3

    print(PureTrainSamples.shape[0], PureValSamples.shape[0], PureTestSamples.shape[0])
    numPureTrain = PureTrainSamples.shape[0]
    numPureTest = PureTestSamples.shape[0]
    numPureVal = PureValSamples.shape[0]

    assert (numPureTrain == 2800)
    assert (numPureTest == 640)
    assert (numPureVal == 400)

    print("Generating smoothed synthetic data...")
    Xtrain_100 = SyntheticGenerator(datapath, trainFiles, 100, 1000, int(num_each/10), True)
    Xtrain_200 = SyntheticGenerator(datapath, trainFiles, 200, 1000, int(num_each/10), True)
    Xtrain_300 = SyntheticGenerator(datapath, trainFiles, 300, 1000, int(num_each/10), True)
    Xtrain_400 = SyntheticGenerator(datapath, trainFiles, 400, 1000, int(num_each/10), True)
    Xtrain_500 = SyntheticGenerator(datapath, trainFiles, 500, 1000, int(num_each/10), True)

    Xtest_100 = SyntheticGenerator(datapath, testFiles, 100, 1000, int(num_each / 10), True)
    Xtest_200 = SyntheticGenerator(datapath, testFiles, 200, 1000, int(num_each / 10), True)
    Xtest_300 = SyntheticGenerator(datapath, testFiles, 300, 1000, int(num_each / 10), True)
    Xtest_400 = SyntheticGenerator(datapath, testFiles, 400, 1000, int(num_each / 10), True)
    Xtest_500 = SyntheticGenerator(datapath, testFiles, 500, 1000, int(num_each / 10), True)

    Xval_100 = SyntheticGenerator(datapath, valFiles, 100, 1000, int(num_each / 10), True)
    Xval_200 = SyntheticGenerator(datapath, valFiles, 200, 1000, int(num_each / 10), True)
    Xval_300 = SyntheticGenerator(datapath, valFiles, 300, 1000, int(num_each / 10), True)
    Xval_400 = SyntheticGenerator(datapath, valFiles, 400, 1000, int(num_each / 10), True)
    Xval_500 = SyntheticGenerator(datapath, valFiles, 500, 1000, int(num_each / 10), True)

    trainX = np.vstack([PureTrainSamples, Xtrain_100, Xtrain_200, Xtrain_300, Xtrain_400, Xtrain_500])
    testX = np.vstack([PureTestSamples, Xtest_100, Xtest_200, Xtest_300, Xtest_400, Xtest_500])
    valX = np.vstack([PureValSamples, Xval_100, Xval_200, Xval_300, Xval_400, Xval_500])
    assert (trainX.shape[0] == 5600)
    assert (testX.shape[0] == 1280)
    assert (valX.shape[0] == 800)
    trainY = np.vstack([np.ones((numPureTrain,1)), np.zeros((numPureTrain,1))])
    testY = np.vstack([np.ones((numPureTest, 1)), np.zeros((numPureTest, 1))])
    valY = np.vstack([np.ones((numPureVal, 1)), np.zeros((numPureVal, 1))])

    np.save(file= out + 'smooth/trainX.npy', arr=trainX)
    np.save(file=out + 'smooth/trainY.npy', arr=trainY)
    np.save(file=out + 'smooth/testX.npy', arr=testX)
    np.save(file=out + 'smooth/testY.npy', arr=testY)
    np.save(file=out + 'smooth/valX.npy', arr=valX)
    np.save(file=out + 'smooth/valY.npy', arr=valY)

    print("Generating non-smoothed synthetic data...")
    Xtrain_100 = SyntheticGenerator(datapath, trainFiles, 100, 1000, int(num_each / 10), False)
    Xtrain_200 = SyntheticGenerator(datapath, trainFiles, 200, 1000, int(num_each / 10), False)
    Xtrain_300 = SyntheticGenerator(datapath, trainFiles, 300, 1000, int(num_each / 10), False)
    Xtrain_400 = SyntheticGenerator(datapath, trainFiles, 400, 1000, int(num_each / 10), False)
    Xtrain_500 = SyntheticGenerator(datapath, trainFiles, 500, 1000, int(num_each / 10), False)

    Xtest_100 = SyntheticGenerator(datapath, testFiles, 100, 1000, int(num_each / 10), False)
    Xtest_200 = SyntheticGenerator(datapath, testFiles, 200, 1000, int(num_each / 10), False)
    Xtest_300 = SyntheticGenerator(datapath, testFiles, 300, 1000, int(num_each / 10), False)
    Xtest_400 = SyntheticGenerator(datapath, testFiles, 400, 1000, int(num_each / 10), False)
    Xtest_500 = SyntheticGenerator(datapath, testFiles, 500, 1000, int(num_each / 10), False)

    Xval_100 = SyntheticGenerator(datapath, valFiles, 100, 1000, int(num_each / 10), False)
    Xval_200 = SyntheticGenerator(datapath, valFiles, 200, 1000, int(num_each / 10), False)
    Xval_300 = SyntheticGenerator(datapath, valFiles, 300, 1000, int(num_each / 10), False)
    Xval_400 = SyntheticGenerator(datapath, valFiles, 400, 1000, int(num_each / 10), False)
    Xval_500 = SyntheticGenerator(datapath, valFiles, 500, 1000, int(num_each / 10), False)

    trainX = np.vstack([PureTrainSamples, Xtrain_100, Xtrain_200, Xtrain_300, Xtrain_400, Xtrain_500])
    testX = np.vstack([PureTestSamples, Xtest_100, Xtest_200, Xtest_300, Xtest_400, Xtest_500])
    valX = np.vstack([PureValSamples, Xval_100, Xval_200, Xval_300, Xval_400, Xval_500])
    assert (trainX.shape[0] == 5600)
    assert (testX.shape[0] == 1280)
    assert (valX.shape[0] == 800)
    trainY = np.vstack([np.ones((numPureTrain, 1)), np.zeros((numPureTrain, 1))])
    testY = np.vstack([np.ones((numPureTest, 1)), np.zeros((numPureTest, 1))])
    valY = np.vstack([np.ones((numPureVal, 1)), np.zeros((numPureVal, 1))])

    np.save(file=out + 'nosmooth/trainX.npy', arr=trainX)
    np.save(file=out + 'nosmooth/trainY.npy', arr=trainY)
    np.save(file=out + 'nosmooth/testX.npy', arr=testX)
    np.save(file=out + 'nosmooth/testY.npy', arr=testY)
    np.save(file=out + 'nosmooth/valX.npy', arr=valX)
    np.save(file=out + 'nosmooth/valY.npy', arr=valY)










