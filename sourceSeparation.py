import matplotlib.pyplot as plt
import torch.optim as optim
import torch 
from matplotlib.pyplot import figure
import numpy as np
import gans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['figure.figsize'] = [16, 8]

def maxlikelihood_separatesources(
    generators,
    discriminators,
    loader_mix, 
    epochs=5,
    doPrint=False
):
    generator1, generator2 = generators
    discriminator1, discriminator2 = discriminators
    inputSize = generator1.inputSize

    mixes = []
    extractedSpikes = []
    extractedNoises = []
    generatedMixes = []
    tError = 0
    for i, [mix] in enumerate(loader_mix):
        if i > 0:
            #we process all windows in parallel
            print("fix batching for separation")
            break 
        mix = mix.squeeze().to(device)
        nmix = mix.size(0)
        print(mix.shape)
        x1 = torch.rand((nmix, inputSize), device=device, requires_grad=True)
        x2 = torch.rand((nmix, inputSize), device=device, requires_grad=True)

        optimizer_sourcesep = optim.Adam([x1, x2], lr=1e-3, betas=(0.5, 0.999))
        for epoch in range(epochs):
            source1 = generator1.forward(x1)
            source2 = generator2.forward(x2)
            mix_sum = source1 +  source2
            serr = torch.mean(torch.abs(source1[1:] - source1[:-1]) \
                              +torch.abs(source2[1:] - source2[:-1]))
            err = torch.mean((mix - mix_sum) ** 2) + 0.01*serr

            tError += err.cpu().detach().numpy()
            
            err.backward()
            optimizer_sourcesep.step()
            x1.grad.data.zero_()
            x2.grad.data.zero_()
        
        extractedSpikes = generator1.forward(x1)
        extractedNoises = generator2.forward(x2)
        generatedMixes = extractedNoises + extractedSpikes
        mixes = mix

    extractedSpikes = [extractedSpike[0:40] + 1j * extractedSpike[40:] for extractedSpike in extractedSpikes]
    extractedSpikes = torch.stack(extractedSpikes)
    extractedSpikes = torch.fft.irfft(extractedSpikes).cpu().detach()

    extractedNoises = [extractedNoise[0:40] + 1j * extractedNoise[40:] for extractedNoise in extractedNoises]
    extractedNoises = torch.stack(extractedNoises)
    extractedNoises = torch.fft.irfft(extractedNoises).cpu().detach()

    if doPrint:
        generatedMixes = [extractedNoise[0:40] + 1j * extractedNoise[40:] for extractedNoise in generatedMixes]
        generatedMixes = torch.stack(generatedMixes)
        generatedMixes = torch.fft.irfft(generatedMixes).cpu().detach()

        mixes = [extractedNoise[0:40] + 1j * extractedNoise[40:] for extractedNoise in mixes]
        mixes = torch.stack(mixes)
        mixes = torch.fft.irfft(mixes).cpu().detach()
        plt.rcParams['figure.figsize'] = [16, 4]
        for i,j in enumerate([0,1]):
            fig, axs = plt.subplots(1,3)
            plt.setp(axs, ylim=(-0.4,1))
            fig.tight_layout()
            
            axs[0].title.set_text('Estimated Source 1')
            axs[1].title.set_text('Estimated Source 2')
            axs[2].title.set_text('Mixture (Blue) vs Sum of estimated sources')

            axs[0].plot(extractedSpikes[j])
            axs[1].plot(extractedNoises[j])
            axs[2].plot(mixes[j])
            axs[2].plot(generatedMixes[j])
            plt.show()

        
    
    return (extractedSpikes, extractedNoises, tError / nmix)