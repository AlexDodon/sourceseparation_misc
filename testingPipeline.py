import numpy as np
import torch
import matplotlib.pyplot as plt

import gans

def confusionMatrix(labels, predictions):
    if max(labels) > 1 or max(predictions) > 1 or min(labels) < 0 or min(predictions) < 0:
        raise Exception("We use only 2 classes: 1 and 0")

    if len(labels) != len(predictions):
        raise Exception("the lengths are not equal")
        
    confusionMatrix = np.zeros((2, 2)).astype("int64")

    for l,p in zip(labels, predictions):
        confusionMatrix[l][p] += 1

    return confusionMatrix

def metrics(confusionMatrix):
    sensitivity = confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[1][0])
    specificity = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[0][1])
    accuracy    = (confusionMatrix[1][1] + confusionMatrix[0][0]) \
        / (confusionMatrix[1][1] + confusionMatrix[0][0] + confusionMatrix[1][0] + confusionMatrix[0][1]) 
    precision   = confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[0][1]) 
    f1          = 2 * ((precision * sensitivity)/(precision + sensitivity))

    return (accuracy, sensitivity, specificity, f1)

def interpretSeparation(extractedSpikes, critic, vallabel, method="energy", test=False, testThreshold=0):
    if method == "energy":
        energy = torch.sum(torch.square(extractedSpikes), 0)

        thresholds =  [x / 20 for x in range(1,600,1)]
        toTest = energy

    if method == "critic":
        extractedSpikes = torch.fft.rfft(extractedSpikes, dim=1)
        extractedSpikes = torch.stack([torch.cat((x.real,x.imag),0) for x in extractedSpikes])
        criticScores = critic.forward(extractedSpikes.to(gans.device)).cpu().detach().numpy()
        criticScores = [x for [x] in criticScores]

        thresholds =  [x / 2 for x in range(-200,0,1)]
        toTest = criticScores


    hist, edges = np.histogram(toTest, bins = 100)

    plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black", align="edge")
    plt.show()

    used = []
    sensitivities = []
    accuracies = []
    specificities = []
    f1s = []

    for threshold in thresholds:
        try:
            res = torch.where(toTest > threshold, 1, 0).numpy().astype("int64")

            confusionMatrix = confusionMatrix(vallabel, res)

            accuracy, sensitivity, specificity, f1 = metrics(confusionMatrix)

            sensitivities.append(sensitivity)
            accuracies.append(accuracy)
            specificities.append(specificity)
            used.append(threshold)
            f1s.append[f1]
        except:
            continue
        
    _, axs = plt.subplots(1,3)
    axs[0].plot(accuracies)
    axs[0].title.set_text("Accuracy")
    axs[1].plot(specificities)
    axs[1].title.set_text("Specificity")
    axs[2].plot(sensitivities)
    axs[2].title.set_text("Sensitivity")
    plt.show()

    threshold = used[f1s.index(max(f1s))]
    print("Threshold for best F1: {}".format(threshold))
    
    res = []

    if test:
        threshold = testThreshold

    print("Threshold: {}".format(threshold))

    res = torch.where(energy > threshold, 1, 0).numpy().astype("int64")

    confusionMatrix = confusionMatrix(vallabel, res)

    accuracy, sensitivity, specificity, f1 = metrics(confusionMatrix)

    print("Sensitivity: {}".format(sensitivity))
    print("Specificity: {}".format(specificity))
    print("Accuracy: {}".format(accuracy))
    

    return (threshold, confusionMatrix, accuracy, sensitivity, specificity)