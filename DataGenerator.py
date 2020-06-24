import pandas as pd
import os
import numpy as np
import random
import pickle

seed = 1
random.seed(seed)
np.random.seed(seed)


datapath = "/home/jxin05/rawdata/"

class x_pair:
    def __init__(self, x1, x2, label): # 1x3x200
        assert(x1.shape == x2.shape)
        self.x1 = x1
        self.x2 = x2
        self.label = label

# split 96 users into 3 groups
# 1 to 70 - training group
# 71 to 80 - validation group
# 81 to 96 - testing group
def train_test_split(users):
    num_user = len(users)
    idx = np.random.permutation(num_user)
    trainIdx = idx[:70]
    valIdx = idx[70:80]
    testIdx = idx[80:]
    trainUsers = []
    testUsers = []
    trainFiles = []
    testFiles = []
    valUsers = []
    valFiles = []
    for i in trainIdx:
        trainUsers.append(users[i])
        trainFiles.append(users[i]+'session1.csv')
        trainFiles.append(users[i]+'session2.csv')
    for i in testIdx:
        testUsers.append(users[i])
        testFiles.append(users[i]+'session1.csv')
        testFiles.append(users[i]+'session2.csv')
    for i in valIdx:
        valUsers.append(users[i])
        valFiles.append(users[i]+'session1.csv')
        valFiles.append(users[i]+'session2.csv')
    return trainUsers, testUsers, valUsers, trainFiles, testFiles, valFiles

def data_split(path):
    file_list = os.listdir(path)
    users = []
    for file in file_list:
        user = file[:-12]
        if user not in users:
            users.append(user)
    trainUsers, testUsers, valUsers, trainFiles, testFiles, valFiles = train_test_split(users)
    if(len(trainUsers)*2 == len(trainFiles) and len(testUsers)*2 == len(testFiles)):
        print("Train, Test, Validation group split done!")
        return trainFiles, testFiles, valFiles
    else:
        print("Some users' data is missing")
        return

# remove some columns from raw .csv file
def remove_col(df):
    return df.drop(['EID','time','time_in_ms'], axis =1)

# sample a portion with certain length from a .csv file
def sample(file, length):
    df = pd.read_csv(file)
    df = remove_col(df)
    idx = np.random.randint(0, len(df))
    while(idx + length >= len(df)):
        idx = np.random.randint(0, len(df))
    res = df.iloc[idx:idx+length].values
    return res



# if two 2s data samples are from the same user, the pair has label 1, otherwise, the pair has label 0
# the function samples num*2 label 1 samples and num*2 label 0 samples from a group,
# can be training, validation or testing group.
# returned X is a dictionary containing two lists
# one list has label 1 pairs and one has label 0 pairs
def samplePairs(files, num, datapath):
    count = 0
    X = {'same': [], 'diff': []}
    while (count < num):
        idx1 = np.random.randint(0, len(files))
        idx2 = np.random.randint(0, len(files))
        if (idx1 == idx2):
            continue
        if (files[idx1][:-12] == files[idx2][:-12]):
            continue
        file1 = datapath + files[idx1]
        file2 = datapath + files[idx2]
        s11 = sample(file1, 200).T
        s12 = sample(file1, 200).T
        s21 = sample(file2, 200).T
        s22 = sample(file2, 200).T

        same_pair1 = x_pair(s11, s12, 1)
        same_pair2 = x_pair(s21, s22, 1)

        diff_pair1 = x_pair(s11, s21, 0)
        diff_pair2 = x_pair(s12, s22, 0)

        X['same'].append(same_pair1)
        X['same'].append(same_pair2)

        X['diff'].append(diff_pair1)
        X['diff'].append(diff_pair2)

        count += 1
    return X


if __name__ == '__main__':
    trainFiles, testFiles, valFiles = data_split(datapath)

    file = open('/home/jxin05/Phase1Data/trainFiles.txt', 'w')
    file.write(str(trainFiles))
    file.close()
    file = open('/home/jxin05/Phase1Data/testFiles.txt', 'w')
    file.write(str(testFiles))
    file.close()
    file = open('/home/jxin05/Phase1Data/validationFiles.txt', 'w')
    file.write(str(valFiles))
    file.close()

    X_train = samplePairs(trainFiles, 4000, datapath)
    X_val = samplePairs(valFiles, 1500, datapath)
    X_test = samplePairs(testFiles, 10, datapath)

    with open('/home/jxin05/Phase1Data/train.pickle', 'wb') as f:
        pickle.dump(X_train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/home/jxin05/Phase1Data/test.pickle', 'wb') as f:
        pickle.dump(X_test, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/home/jxin05/Phase1Data/validation.pickle', 'wb') as f:
        pickle.dump(X_val, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Phase1 train data generated successfully!")
