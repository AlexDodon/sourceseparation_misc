import importlib
import numpy as np
import matplotlib.pyplot as plt
from models import *
from preprocessing import *

importlib.import_module("models")
importlib.import_module("preprocessing")

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