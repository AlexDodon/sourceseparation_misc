import math
import torch
import numpy as np

def computeThreshold(*datasets):
    s = 0
    c = 0

    for dataset in datasets:
        (data,) = dataset.dataset[:]

        #sumeaza elementele in 2d => o valoare 
        s += torch.sum(data)
        #numara batchuri * window size
        c += data.shape[0] * data[0].shape[0]

    mean = s.numpy()/c

    s = 0

    for dataset in datasets:
        (data,) = dataset.dataset[:]
        s += torch.sum(torch.square(data - mean))

    # we use c - 1 because we do not have the whole population
    sd = math.sqrt(s.numpy()/ (c-1))

    return (mean, sd)

def thresholdDatasets(mean, sd, *datasets):
    predictions = np.array([])

    for dataset in datasets:
        (data,) = dataset.dataset[:]

        newPredictions = torch.where((data >= mean + sd).any(1), 1, 0).numpy()

        predictions = np.concatenate([predictions, newPredictions], axis=0)

    return predictions.astype("int64")


def testThresholding(minMul, maxMul, trainSpikes, valSpikes, testSpikes, trainNoise, valNoise, testNoise):

    mean, sd = computeThreshold(trainSpikes, valSpikes, testSpikes, trainNoise, valNoise, testNoise)    

    res = {}

    for sdMultiple in range(minMul, maxMul + 1):

        testPredictions = thresholdDatasets(mean, sd * sdMultiple, testSpikes, testNoise)

        testlabel = np.concatenate((np.ones(testSpikes.dataset[:][0].shape[0]), np.zeros(testNoise.dataset[:][0].shape[0])), axis=0)

        res[sdMultiple] = [testlabel.astype("int64"), testPredictions]

    return res