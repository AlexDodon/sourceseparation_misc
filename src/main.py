import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def checkSpikeLocations():
    gt = scipy.io.loadmat('../../data/gen/ground_truth.mat')
    sim = scipy.io.loadmat('../../data/gen/simulation_1.mat')
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
    gt = scipy.io.loadmat('../../data/gen/ground_truth.mat')
    sim = scipy.io.loadmat('../../data/gen/simulation_{}.mat'.format(simNo))

    data = sim["data"][0]
    firstSamples = gt["spike_first_sample"][0][simNo - 1][0]

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
                spikes.append(data[simIndex : simIndex + spikeLength])
                simIndex += spikeLength
                spikeIndex += 1
        else: # No more spikes; might still have hash
            hash.append(data[simIndex:])
            simIndex = sampleNumber

    return (np.array(spikes, dtype=object), np.array(hash, dtype=object))


def main():
    spikes, hash = splitSim(1)
    print(spikes.shape)
    print(hash.shape)
    hashSamples = 0

    for h in hash:
        hashSamples += len(h)
    print(hashSamples)

    fig, axs = plt.subplots(2)
    axs[0].plot(spikes[0])
    axs[1].plot(hash[0])

    plt.show()
    

if __name__ == "__main__":
    main()