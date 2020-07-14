# Extension-Work-for-IoT-CIS700

TrainNet.py trains a SimameseNet in two phases: supervised learning and multiple instance learning

DataGenerator.py split 96 users into 3 groups and generate phase 1 training data

SiameseNet.py defines different network structures 

Phase 1 training is to train a Siamese net to distinguish data from different users. It helps the model to converage when doing MIL.

MIL1 is about using MIL-Siamese model to detect non-pure data

MIL2 is about using MIL-Siamese model to authenticate users given a small template of the genuine user

MIL2Baseline_authenticate10s.py wants to build a baseline model only using Siamese Network to compare 2 accelerometer readings with length 10s, so that authentication can be done. Siamese10s-Evaluation.py evaluates the baseline model on differetn categories of data: pure data from genuine user, pure data from imposters and synthetica data with different attack levels.
