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
    for i, [mix] in enumerate(loader_mix): 
        mix = mix.squeeze().view(-1,1,79).to(device)
        print('Processing source ',i)
        Nmix = mix.size(0)
        x1 = Variable(torch.rand(Nmix,1,inputSize, device=device), requires_grad=True)
        x2 = Variable(torch.rand(Nmix,1,inputSize, device=device), requires_grad=True)

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

        x1hat.append(generator1.forward(x1).data.cpu().numpy())
        x2hat.append(generator2.forward(x2).data.cpu().numpy())
        mixes.append(mix.cpu().numpy())
    
    _, axs = plt.subplots(4,3)
    
    for i in range(0,4):
        axs[i][0].plot(x1hat[i][0][0])
        axs[i][0].title.set_text('Estimated Source 1')
        axs[i][1].plot(x2hat[i][0][0])
        axs[i][1].title.set_text('Estimated Source 2')
        axs[i][2].plot(mixes[i][0][0])
        axs[i][2].title.set_text('Mixture')

    plt.show()
    
    return x1hat

# some interpretation of results

import numpy as np
import torch
import matplotlib.pyplot as plt


extractedSpikesValidation = [ x for [[x]] in cleanextractedSpikesValidation]
extractedSpikesValidation = np.array(extractedSpikesValidation)
energy = []

for extractedSpike in extractedSpikesValidation:
    energy.append(sum(np.power(extractedSpike, 2)))

energy = np.array(energy)
hist, edges = np.histogram(energy, bins = 30)

plt.plot(np.sort(energy))
plt.show()
plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black", align="edge")
plt.show()

tresholds =  [x / 10 for x in range(125,155,1)]

precisions = []
recalls = []
f1s = []
accs = []

for treshold in tresholds:
    res = []

    for elem in energy:
    if elem > treshold:
        res.append(1)
    else:
        res.append(0)
    
    truepos = 0
    falsepos = 0
    trueneg = 0
    falseneg = 0

    for i in range(0, len(vallabel)):
        if (
            vallabel[i] == 1 or 
            vallabel[i - 1] == 1 or 
            vallabel[i - 2] == 1 or 
            i + 1 < 180 and vallabel[i + 1] == 1 or 
            i + 2 < 180 and vallabel[i + 2] == 1
        ):
            if res[i] == 1:
                truepos += 1
            else:
                falseneg += 1
        else:
            if res[i] == 1:
                falsepos += 1
            else:
                trueneg += 1

    precision = truepos / (truepos + falsepos)
    recall = truepos / (truepos + falseneg)  
    f1 = 2 * ((precision * recall)/(precision+recall)))
    acc = (truepos + trueneg) / len(vallabel))
    
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    accs.append(acc)
    
_, axs = plt.subplots(1,2)

axs[0].plot(tresholds)
axs[1].plot(accs)
plt.show()
