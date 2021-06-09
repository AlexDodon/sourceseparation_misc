import math

def computeThreshold(*datasets):
    s = 0
    c = 0

    for dataset in datasets:
        for [batch] in dataset:
            for window in batch:
                s += sum(window)
                c += len(window)

    mean = s.numpy()/c

    s = 0

    for dataset in datasets:
        for [batch] in dataset:
            for window in batch:
                s += sum([(x - mean) ** 2 for x in window])
    
    sd = math.sqrt(s/c)

    return (mean, sd)

def thresholdDatasets(mean, sd, *datasets):
    labels = []

    for dataset in datasets:
        for [batch] in dataset:
            for window in batch:
                if max(window) >= mean + sd:
                    labels.append(1)
                else:
                    labels.append(0)

    return labels