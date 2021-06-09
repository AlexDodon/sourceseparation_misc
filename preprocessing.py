import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data_utils
import torch 
import math

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

def splitSim(simNo, valRatio, testRatio, inclusionThreshold=0.4):
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
                # right now were taking them into consideration
                #if spikeClasses[spikeIndex] != 0: # It isn't a multi unit spike
                if max(data[simIndex : simIndex + spikeLength]) >= inclusionThreshold:
                    spikes.append(data[simIndex : simIndex + spikeLength])
                
                simIndex += spikeLength
                spikeIndex += 1
        else: # No more spikes; might still have hash
            hash.append(data[simIndex:])
            simIndex = sampleNumber

    background = []
    drown = []
    
    for chunk in hash:
        if len(background) == len(spikes) and len(drown) == len(spikes):
            break

        i = 0
        sample = []
        while i < len(chunk):
            sample.append(chunk[i])
            i += 1
            if i % 79 == 0 and (len(background) < len(spikes) or len(drown) < len(spikes)):
                if len(background) > len(drown):
                    drown.append(sample)
                else:
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

    drown = torch.Tensor(drown).to(device)
    drown = torch.fft.rfft(drown, dim=1)
    drown = torch.stack([torch.cat((x.real,x.imag),0) for x in drown])
    drown = drown.cpu().numpy()

    rng.shuffle(background)
    rng.shuffle(spikes)
    rng.shuffle(drown)

    trainSpikes = []
    valSpikes = []
    testSpikes = []
    trainBg = []
    valBg = []
    testBg = []
    trainDrown = []
    valDrown = []
    testDrown = []

    l = len(spikes)
    for i in range(l):
        if i <= l * (1 - valRatio - testRatio):
            trainSpikes.append(spikes[i])
            trainBg.append(background[i])
            trainDrown.append(drown[i])
        elif i <= l * (1 - testRatio):
            valSpikes.append(spikes[i])
            valBg.append(background[i])
            valDrown.append(drown[i])
        else:
            testSpikes.append(spikes[i])
            testBg.append(background[i])
            testDrown.append(drown[i])

    return (trainSpikes, valSpikes, testSpikes, trainBg, valBg, testBg, trainDrown, valDrown, testDrown)

def drownSpikes(snr, trainSpikes, valSpikes, testSpikes, trainDrown, valDrown, testDrown):
    scale = []
    trainRes = []
    valRes = []
    testRes = []
    valPrint = []

    for i,(s,n) in enumerate(zip(trainSpikes, trainDrown)):
        try:
            sc = math.sqrt(snr * (sum([a**2 for a in n]) / sum(a**2 for a in s)))
        except:
            continue
        scale.append(sc)
        trainRes.append(sc * s + n)

    for i,(s,n) in enumerate(zip(valSpikes, valDrown)):
        try:
            sc = math.sqrt(snr * (sum([a**2 for a in n]) / sum(a**2 for a in s)))
        except:
            continue
        scale.append(sc)
        valRes.append(sc * s + n)
        valPrint.append(s)

    for i,(s,n) in enumerate(zip(testSpikes, testDrown)):
        try:
            sc = math.sqrt(snr * (sum([a**2 for a in n]) / sum(a**2 for a in s)))
        except:
            continue
        scale.append(sc)
        testRes.append(sc * s + n)

    plt.rcParams['figure.figsize'] = [16, 16]

    print(f"For snr {snr}")
    print(len(scale))

    hist, edges = np.histogram(np.array(scale), bins = 100)
    plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black", align="edge")
    #plt.plot(np.sort(scale))
    plt.show()

    fig, axs = plt.subplots(8,4)
    plt.setp(axs, ylim=(-0.4,1.5))
    fig.tight_layout()

    for i,(a,b) in enumerate(zip(valPrint[:32], valRes[:32])):  
        axs[i//4][i%4].plot(torch.fft.irfft(torch.from_numpy(a)[0:40] + 1j * torch.from_numpy(a)[40:]).cpu().detach().numpy())
        axs[i//4][i%4].plot(torch.fft.irfft(torch.from_numpy(b)[0:40] + 1j * torch.from_numpy(b)[40:]).cpu().detach().numpy())
    plt.show()


    plt.rcParams['figure.figsize'] = [16, 8]

    return (np.array(trainRes), np.array(valRes), np.array(testRes))

def gen_loaders(batchsize, includedSimulations, valRatio=0.1, testRatio=0.1, doDrown=False, snr=10, inclusionThreshold=0.4):
    trainSpikes = np.empty_like(np.ones(80).reshape(1,80))
    valSpikes = np.empty_like(np.ones(80).reshape(1,80))
    testSpikes = np.empty_like(np.ones(80).reshape(1,80))
    trainBg = np.empty_like(np.ones(80).reshape(1,80))
    valBg = np.empty_like(np.ones(80).reshape(1,80))
    testBg = np.empty_like(np.ones(80).reshape(1,80))
    trainDrown = np.empty_like(np.ones(80).reshape(1,80))
    valDrown = np.empty_like(np.ones(80).reshape(1,80))
    testDrown = np.empty_like(np.ones(80).reshape(1,80))

    for i in range(1,includedSimulations + 1):
        print(f"Simularea {i}")
        trs, vs,ts, trb, vb, tb, trd, vd, td = splitSim(i, valRatio, testRatio, inclusionThreshold)
        trainSpikes = np.concatenate((trainSpikes,trs),axis=0)
        valSpikes = np.concatenate((valSpikes,vs),axis=0)
        testSpikes = np.concatenate((testSpikes,ts),axis=0)
        trainBg = np.concatenate((trainBg,trb),axis=0)
        valBg = np.concatenate((valBg,vb),axis=0)
        testBg = np.concatenate((testBg,tb),axis=0)
        trainDrown = np.concatenate((trainDrown,trd),axis=0)
        valDrown = np.concatenate((valDrown,vd),axis=0)
        testDrown = np.concatenate((testDrown,td),axis=0)

    valSize = len(valSpikes)
    testSize = len(testSpikes)

    if doDrown:
        trainSpikes, valSpikes, testSpikes = drownSpikes(snr, trainSpikes, valSpikes, testSpikes, trainDrown, valDrown, testDrown)

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