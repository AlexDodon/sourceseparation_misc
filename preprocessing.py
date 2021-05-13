import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data_utils
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['figure.figsize'] = [16, 8]

def checkSpikeLocations():
    gt = scipy.io.loadmat('../data/gen/ground_truth.mat')
    sim = scipy.io.loadmat('../data/gen/simulation_1.mat')
    print(gt["spike_first_sample"][0][0].shape)
    print(gt["su_waveforms"][0][0].shape)
    print(gt["spike_classes"][0][0].shape)


    print(gt["spike_first_sample"][0][0][0][0])
    print(gt["spike_classes"][0][0][0][1])

    first_spike_start = gt["spike_first_sample"][0][0][0][0]
    first_spike_class = gt["spike_classes"][0][0][0][1]

    data = sim["data"][0][first_spike_start : first_spike_start + 80]

    waveform = gt["su_waveforms"][0][0][first_spike_class]

    fig, axs = plt.subplots(2)
    fig.suptitle('Single unit waveform vs. Simulated signal')
    axs[0].plot(range(0,316), waveform)
    axs[1].plot(range(0,80), data)

    plt.show()

def splitSim(simNo, valRatio, testRatio):
    gt = scipy.io.loadmat('../data/gen/ground_truth.mat')
    sim = scipy.io.loadmat('../data/gen/simulation_{}.mat'.format(simNo))

    data = sim["data"][0]
    dataMax = max(data)
    dataMin = min(data)
    div = dataMax - dataMin

    #data = [(x - dataMin) / div for x in data]

    firstSamples = gt["spike_first_sample"][0][simNo - 1][0]
    spikeClasses = gt["spike_classes"][0][simNo - 1][0]

    spikes = []
    hash = []

    simIndex = 0
    spikeIndex = 0
    sampleNumber = len(data)
    spikeNumber = len(firstSamples)
    spikeLength = 79 # 316 at 96k Hz downsampled to 24k Hz

    while simIndex < sampleNumber:
        if spikeIndex < spikeNumber: # I still have spikes 
            if simIndex < firstSamples[spikeIndex]: # a non-spike interfal follows
                hash.append(data[simIndex : firstSamples[spikeIndex]])
                simIndex = firstSamples[spikeIndex]
            else: # I have a spike
                # TODO consider taking into consideration multi unit spikes
                if spikeClasses[spikeIndex] != 0: # It isn't a multi unit spike
                    spikes.append(data[simIndex : simIndex + spikeLength])
                    simIndex += spikeLength
                else:
                    simIndex += spikeLength + 30 # it is a multi unit spike
                spikeIndex += 1
        else: # No more spikes; might still have hash
            hash.append(data[simIndex:])
            simIndex = sampleNumber

    background = []
    
    for chunk in hash:
        if len(background) == len(spikes):
            break

        i = 0
        sample = []
        while i < len(chunk):
            sample.append(chunk[i])
            i += 1
            if i % 79 == 0 and len(background) < len(spikes):
                background.append(sample)
                sample = []
                
    rng = np.random.default_rng()

    spikes = torch.Tensor(spikes).to(device)
    spikes = torch.fft.rfft(spikes, dim=1)
    spikes = torch.stack([torch.cat((x.real,x.imag),0) for x in spikes])
    spikes = spikes.cpu().numpy()

    background = torch.Tensor(background).to(device)
    background = torch.fft.rfft(background, dim=1)
    background = torch.stack([torch.cat((x.real,x.imag),0) for x in background])
    background = background.cpu().numpy()

    rng.shuffle(background)
    rng.shuffle(spikes)

    trainSpikes = []
    valSpikes = []
    testSpikes = []
    trainBg = []
    valBg = []
    testBg = []

    l = len(spikes)
    for i in range(l):
        if i <= l * (1 - valRatio - testRatio):
            trainSpikes.append(spikes[i])
            trainBg.append(background[i])
        elif i <= l * (1 - testRatio):
            valSpikes.append(spikes[i])
            valBg.append(background[i])
        else:
            testSpikes.append(spikes[i])
            testBg.append(background[i])

    return (trainSpikes, valSpikes, testSpikes, trainBg, valBg, testBg)

def gen_loaders(batchsize, includedSimulations, valRatio=0.1, testRatio=0.1):
    trainSpikes = np.empty_like(np.ones(80).reshape(1,80))
    valSpikes = np.empty_like(np.ones(80).reshape(1,80))
    testSpikes = np.empty_like(np.ones(80).reshape(1,80))
    trainBg = np.empty_like(np.ones(80).reshape(1,80))
    valBg = np.empty_like(np.ones(80).reshape(1,80))
    testBg = np.empty_like(np.ones(80).reshape(1,80))

    for i in range(1,includedSimulations + 1):
        trs, vs,ts, trb, vb, tb = splitSim(i, valRatio, testRatio)
        trainSpikes = np.concatenate((trainSpikes,trs),axis=0)
        valSpikes = np.concatenate((valSpikes,vs),axis=0)
        testSpikes = np.concatenate((testSpikes,ts),axis=0)
        trainBg = np.concatenate((trainBg,trb),axis=0)
        valBg = np.concatenate((valBg,vb),axis=0)
        testBg = np.concatenate((testBg,tb),axis=0)

    valSize = len(valSpikes)
    testSize = len(testSpikes)

    trainSpikes = data_utils.TensorDataset(torch.from_numpy(trainSpikes[1:]).float())
    valSpikes = data_utils.TensorDataset(torch.from_numpy(valSpikes[1:]).float())
    testSpikes = data_utils.TensorDataset(torch.from_numpy(testSpikes[1:]).float())
    trainBg = data_utils.TensorDataset(torch.from_numpy(trainBg[1:]).float())
    valBg = data_utils.TensorDataset(torch.from_numpy(valBg[1:]).float())
    testBg = data_utils.TensorDataset(torch.from_numpy(testBg[1:]).float())

    trainSpikesLoader = data_utils.DataLoader(trainSpikes, batch_size=batchsize, shuffle=False)
    valSpikesLoader = data_utils.DataLoader(valSpikes, batch_size=valSize, shuffle=False)
    testSpikesLoader = data_utils.DataLoader(testSpikes, batch_size=testSize, shuffle=False)
    trainBgLoader = data_utils.DataLoader(trainBg, batch_size=batchsize, shuffle=False)
    valBgLoader = data_utils.DataLoader(valBg, batch_size=valSize, shuffle=False)
    testBgLoader = data_utils.DataLoader(testBg, batch_size=testSize, shuffle=False)
    
    return (trainSpikesLoader, valSpikesLoader, testSpikesLoader, trainBgLoader, valBgLoader, testBgLoader)