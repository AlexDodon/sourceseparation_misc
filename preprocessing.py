import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data_utils
import torch 

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

def splitSim(simNo):
    gt = scipy.io.loadmat('../data/gen/ground_truth.mat')
    sim = scipy.io.loadmat('../data/gen/simulation_{}.mat'.format(simNo))

    data = sim["data"][0]
    dataMax = max(data)
    dataMin = min(data)
    div = dataMax - dataMin

    data = [(x - dataMin) / div for x in data]

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

    return (np.array(spikes), np.array(hash, dtype=object))

def gen_loaders(L1, batchsize):
    
    spikes, bg = splitSim(1)
    
    background = []
    
    for chunk in bg:
        i = 0
        sample = []
        while i < len(chunk):
            sample.append(chunk[i])
            i += 1
            if i % 79 == 0:
                background.append(sample)
                sample = []
                
    background = np.array(background)

    
    
    gt = scipy.io.loadmat('../data/gen/ground_truth.mat')
    
    testsim = scipy.io.loadmat('../data/gen/simulation_11.mat')
    testdata = testsim["data"][0][0:20000]
    testfirstSamples = gt["spike_first_sample"][0][11 - 1][0]
    valsim = scipy.io.loadmat('../data/gen/simulation_12.mat')
    valdata = valsim["data"][0][0:20000]
    valfirstSamples = gt["spike_first_sample"][0][12 - 1][0]

    vd = []
    vl = []
    spikeIndex = 0
    for index in range(0, len(valdata) - 79):
        vd.append(valdata[index:index + 79])
        if index != valfirstSamples[spikeIndex]:
            vl.append(0)
        else:
            vl.append(1)
            spikeIndex += 1
            
    valdata = []
    vallabel = []
    offset = 2
    for i,x in enumerate(vl):
        if x == 1:
            for j in range(i - offset, i):
                valdata.append(vd[j])
                vallabel.append(0)
                
            valdata.append(vd[i])
            vallabel.append(1)
            
            for j in range(i + 1, i + offset + 1):
                valdata.append(vd[j])
                vallabel.append(0)
            
            for j in range(i + 100, i + 100 + 4):
                valdata.append(vd[j])
                vallabel.append(0)
            
    valdata = np.array(valdata)
    vallabel = np.array(vallabel)
            
    td = []
    tl = []
    spikeIndex = 0
    for index in range(0, len(testdata) - 79):
        td.append(testdata[index:index + 79])
        if index != testfirstSamples[spikeIndex]:
            tl.append(0)
        else:
            tl.append(1)
            spikeIndex += 1
            
            
            
    testdata = []
    testlabel = []
    for i,x in enumerate(tl):
        if x == 1:
            for j in range(i - offset, i):
                testdata.append(td[j])
                testlabel.append(0)
                
            testdata.append(td[i])
            testlabel.append(1)
            
            for j in range(i + 1, i + offset + 1):
                testdata.append(td[j])
                testlabel.append(0)
            
            for j in range(i + 100, i + 100 + 4):
                testdata.append(td[j])
                testlabel.append(0)
            
    testdata = np.array(testdata)
    testlabel = np.array(testlabel)
    
    n1 = torch.randn(len(spikes), L1) 
    n2 = torch.randn(len(background), L1) 
    n1 = data_utils.TensorDataset(n1)
    n2 = data_utils.TensorDataset(n2)  
    
    spikes = torch.from_numpy(spikes).float()
    background = torch.from_numpy(background).float()
    valdata = torch.from_numpy(valdata).float()
    testdata = torch.from_numpy(testdata).float()
    
    sim_val        = data_utils.TensorDataset(valdata)
    sim_test        = data_utils.TensorDataset(testdata)
    spike_dataset      = data_utils.TensorDataset(spikes)
    background_dataset = data_utils.TensorDataset(background)
    
    noise1 = data_utils.DataLoader(n1, batch_size=batchsize, shuffle=False)
    noise2 = data_utils.DataLoader(n2, batch_size=batchsize, shuffle=False)
    loader1 = data_utils.DataLoader(spike_dataset, batch_size=batchsize, shuffle=False)
    loader2 = data_utils.DataLoader(background_dataset, batch_size=batchsize, shuffle=False)
    loader_mix_test = data_utils.DataLoader(sim_test, shuffle=False)
    loader_mix_val = data_utils.DataLoader(sim_val, shuffle=False)

    return noise1, noise2, loader1, loader2, loader_mix_val, vallabel, loader_mix_test, testlabel