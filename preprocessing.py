import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data_utils
import torch 
import pickle

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

def splitSim(simNo, trainRatio, valRatio, inclusionThreshold=0.4):
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
    spikes = torch.cat((torch.real(spikes), torch.imag(spikes)), axis=1)
    spikes = spikes.cpu().numpy()

    background = torch.Tensor(background).to(device)
    background = torch.fft.rfft(background, dim=1)
    background = torch.cat((torch.real(background), torch.imag(background)), axis=1)
    background = background.cpu().numpy()

    drown = torch.Tensor(drown).to(device)
    drown = torch.fft.rfft(drown, dim=1)
    drown = torch.cat((torch.real(drown), torch.imag(drown)), axis=1)
    drown = drown.cpu().numpy()

    rng.shuffle(background)
    rng.shuffle(spikes)
    rng.shuffle(drown)

    trainSpikes, valSpikes, testSpikes = np.split(spikes, [int(len(spikes) * trainRatio), int(len(spikes) * (trainRatio + valRatio))])
    trainBg, valBg, testBg = np.split(background, [int(len(background) * trainRatio), int(len(background) * (trainRatio + valRatio))])
    trainDrown, valDrown, testDrown = np.split(drown, [int(len(drown) * trainRatio), int(len(drown) * (trainRatio + valRatio))])

    return (trainSpikes, valSpikes, testSpikes, trainBg, valBg, testBg, trainDrown, valDrown, testDrown)

def drownSpikes(snr, inSpikes,  inDown, doPrint):
    scale = []
    outSpikes = []
    printSpikes = []
    if len(inSpikes) != 0:
        for s,n in zip(inSpikes, inDown):
            try:
                sc = np.sqrt(snr * np.sum(np.square(n)) / np.sum(np.square(s)))
            except:
                continue
            if np.isnan(sc):
                continue
            scale.append(sc)
            printSpikes.append(s)
            outSpikes.append(sc * s + n)

    if doPrint:

        plt.rcParams['figure.figsize'] = [8, 8]
        
        hist, edges = np.histogram(np.array(scale), bins = 100)
        plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black", align="edge")
        plt.title(f"Scaling factor distribution for snr {snr}", fontsize=14)
        plt.show()

        fig, axs = plt.subplots(2,2)
        plt.setp(axs, ylim=(-0.4,1.5))
        fig.suptitle(f"For snr {snr}", fontsize=14)
        fig.tight_layout()
        for i,(a,b) in enumerate(zip(printSpikes[:4], outSpikes[:4])):  
            axs[i//2][i%2].plot(torch.fft.irfft(torch.from_numpy(a)[0:40] + 1j * torch.from_numpy(a)[40:]).cpu().detach().numpy(), label="Original Spike")
            axs[i//2][i%2].plot(torch.fft.irfft(torch.from_numpy(b)[0:40] + 1j * torch.from_numpy(b)[40:]).cpu().detach().numpy(), label="Drowned Spike")

        axs[0][0].legend()
        axs[0][1].legend()
        axs[1][0].legend()
        axs[1][1].legend()
        plt.show()
        plt.rcParams['figure.figsize'] = [16, 8]

    return np.array(outSpikes)

def gen_loaders(batchsize, includedSimulations, trainRatio=0.8, valRatio=0.1, doDrown=False, snr=10, inclusionThreshold=0.4, doPrint=False):
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
        print(f"{i},", end=" ")
        trs, vs,ts, trb, vb, tb, trd, vd, td = splitSim(i, trainRatio, valRatio, inclusionThreshold)
        trainSpikes = np.concatenate((trainSpikes,trs),axis=0)
        valSpikes = np.concatenate((valSpikes,vs),axis=0)
        testSpikes = np.concatenate((testSpikes,ts),axis=0)
        trainBg = np.concatenate((trainBg,trb),axis=0)
        valBg = np.concatenate((valBg,vb),axis=0)
        testBg = np.concatenate((testBg,tb),axis=0)
        trainDrown = np.concatenate((trainDrown,trd),axis=0)
        valDrown = np.concatenate((valDrown,vd),axis=0)
        testDrown = np.concatenate((testDrown,td),axis=0)
    
    print()

    valSize = len(valSpikes)
    testSize = len(testSpikes)

    if doDrown:
        trainSpikes = drownSpikes(snr, trainSpikes[1:], trainDrown[1:], doPrint)
        valSpikes = drownSpikes(snr, valSpikes[1:], valDrown[1:], doPrint)
        testSpikes = drownSpikes(snr, testSpikes[1:], testDrown[1:], doPrint)

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

def it_genLoaders(batchsize, includedSimulations, trainRatio=0.8, valRatio=0.1, snr=10, inclusionThreshold=0.4, doPrint=False, iterations=[1,2,3,4,5], snrs=[1,2,3,4,5,6], baseDatasetPath="../data/datasets"):

    for iteration in iterations:
        print(f"Generating dataset for iteration {iteration}\nProcessing Simulation: ", end="")

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
            print(f"{i},", end=" ")
            trs, vs,ts, trb, vb, tb, trd, vd, td = splitSim(i, trainRatio, valRatio, inclusionThreshold)
            trainSpikes = np.concatenate((trainSpikes,trs),axis=0)
            valSpikes = np.concatenate((valSpikes,vs),axis=0)
            testSpikes = np.concatenate((testSpikes,ts),axis=0)
            trainBg = np.concatenate((trainBg,trb),axis=0)
            valBg = np.concatenate((valBg,vb),axis=0)
            testBg = np.concatenate((testBg,tb),axis=0)
            trainDrown = np.concatenate((trainDrown,trd),axis=0)
            valDrown = np.concatenate((valDrown,vd),axis=0)
            testDrown = np.concatenate((testDrown,td),axis=0)
        
        print()

        valSize = len(valSpikes)
        testSize = len(testSpikes)

        print("Generating non-drowned segment of dataset")

        trainSpikes = data_utils.TensorDataset(torch.from_numpy(trainSpikes[1:]).float())
        trainBg = data_utils.TensorDataset(torch.from_numpy(trainBg[1:]).float())
        
        trainSpikesLoader = data_utils.DataLoader(trainSpikes, batch_size=batchsize, shuffle=False)
        trainBgLoader = data_utils.DataLoader(trainBg, batch_size=batchsize, shuffle=False)

        pickle.dump(trainSpikesLoader, open(f"{baseDatasetPath}/it-{iteration}-non-drowned-trainSpikesLoader.pickle", "wb"))
        pickle.dump(trainBgLoader, open(f"{baseDatasetPath}/it-{iteration}-non-drowned-trainBgLoader.pickle", "wb"))    
        
        valBg = data_utils.TensorDataset(torch.from_numpy(valBg[1:]).float())
        testBg = data_utils.TensorDataset(torch.from_numpy(testBg[1:]).float()) 
        
        valBgLoader = data_utils.DataLoader(valBg, batch_size=valSize, shuffle=False)
        testBgLoader = data_utils.DataLoader(testBg, batch_size=testSize, shuffle=False)   

        pickle.dump(valBgLoader, open(f"{baseDatasetPath}/it-{iteration}-non-drowned-valBgLoader.pickle", "wb"))
        pickle.dump(testBgLoader, open(f"{baseDatasetPath}/it-{iteration}-non-drowned-testBgLoader.pickle", "wb"))

        for snr in snrs:
            print(f"Generating dataset with snr {snr}")
            resValSpikes = drownSpikes(snr, valSpikes[1:], valDrown[1:], doPrint)
            resTestSpikes = drownSpikes(snr, testSpikes[1:], testDrown[1:], doPrint)

            resValSpikes = data_utils.TensorDataset(torch.from_numpy(resValSpikes).float())
            resTestSpikes = data_utils.TensorDataset(torch.from_numpy(resTestSpikes).float())

            resValSpikesLoader = data_utils.DataLoader(resValSpikes, batch_size=valSize, shuffle=False)
            resTestSpikesLoader = data_utils.DataLoader(resTestSpikes, batch_size=testSize, shuffle=False)

            pickle.dump(resValSpikesLoader, open(f"{baseDatasetPath}/it-{iteration}-snr-{snr}-valSpikesLoader.pickle", "wb"))
            pickle.dump(resTestSpikesLoader, open(f"{baseDatasetPath}/it-{iteration}-snr-{snr}-testSpikesLoader.pickle", "wb"))
