import matplotlib.pyplot as plt
import torch.optim as optim
import torch 
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def maxlikelihood_separatesources(
    generators,
    loader_mix, 
    epochs=5
):
    generator1, generator2 = generators
    inputSize = generator1.inputSize

    x1hat, x2hat = [], []
    mixes = []
    extractedSpikes = []
    extractedNoises = []
    for i, [mix] in enumerate(loader_mix): 
        mix = mix.squeeze().to(device)
        print('Processing source ',i)
        x1 = Variable(torch.rand(inputSize, device=device), requires_grad=True)
        x2 = Variable(torch.rand(inputSize, device=device), requires_grad=True)

        optimizer_sourcesep = optim.Adam([x1, x2], lr=1e-3, betas=(0.5, 0.999))
        for epoch in range(epochs):
           
            mix_sum = generator1.forward(x1) + generator2.forward(x2) 
            #Poisson
            eps = 1e-20
            err = torch.mean(-Variable(mix)*torch.log(mix_sum+eps) + mix_sum)

            err.backward()

            optimizer_sourcesep.step()

            x1.grad.data.zero_()
            x2.grad.data.zero_()
        
        extractedSpikes.append(generator1.forward(x1))
        extractedNoises.append(generator2.forward(x2))
        mixes.append(mix)

    extractedSpikes = [extractedSpike[0:40] + 1j * extractedSpike[40:] for extractedSpike in extractedSpikes]
    extractedSpikes = torch.stack(extractedSpikes)
    extractedSpikes = torch.fft.irfft(extractedSpikes).cpu().detach().numpy()

    extractedNoises = [extractedNoise[0:40] + 1j * extractedNoise[40:] for extractedNoise in extractedNoises]
    extractedNoises = torch.stack(extractedNoises)
    extractedNoises = torch.fft.irfft(extractedNoises).cpu().detach().numpy()

    mixes = [extractedNoise[0:40] + 1j * extractedNoise[40:] for extractedNoise in mixes]
    mixes = torch.stack(mixes)
    mixes = torch.fft.irfft(mixes).cpu().detach().numpy()

    _, axs = plt.subplots(4,3)
    plt.setp(axs, ylim=(-0.5,1.5))
    
    for i,j in enumerate([0,7,15,23]):
        axs[i][0].plot(extractedSpikes[j])
        axs[i][0].title.set_text('Estimated Source 1')
        axs[i][1].plot(extractedNoises[j])
        axs[i][1].title.set_text('Estimated Source 2')
        axs[i][2].plot(mixes[j])
        axs[i][2].title.set_text('Mixture')

    plt.show()
    
    return (extractedSpikes, extractedNoises)