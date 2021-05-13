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
    loader_mix, 
    epochs=5
):
    generator1, generator2 = generators
    inputSize = generator1.inputSize

    mixes = []
    extractedSpikes = []
    extractedNoises = []
    generatedMixes = []

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
           
            mix_sum = generator1.forward(x1) + generator2.forward(x2) 
            # #Poisson
            #eps = 1e-20
            #err = torch.mean(-mix*torch.log(mix_sum+eps) + mix_sum)

            # Euclidean
            err = torch.mean((mix - mix_sum) ** 2)

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
    extractedSpikes = torch.fft.irfft(extractedSpikes).cpu().detach().numpy()

    extractedNoises = [extractedNoise[0:40] + 1j * extractedNoise[40:] for extractedNoise in extractedNoises]
    extractedNoises = torch.stack(extractedNoises)
    extractedNoises = torch.fft.irfft(extractedNoises).cpu().detach().numpy()

    generatedMixes = [extractedNoise[0:40] + 1j * extractedNoise[40:] for extractedNoise in generatedMixes]
    generatedMixes = torch.stack(generatedMixes)
    generatedMixes = torch.fft.irfft(generatedMixes).cpu().detach().numpy()

    mixes = [extractedNoise[0:40] + 1j * extractedNoise[40:] for extractedNoise in mixes]
    mixes = torch.stack(mixes)
    mixes = torch.fft.irfft(mixes).cpu().detach().numpy()

    fig, axs = plt.subplots(4,3)
    plt.setp(axs, ylim=(-0.6,1.7))
    fig.tight_layout()
    
    axs[0][0].title.set_text('Estimated Source 1')
    axs[0][1].title.set_text('Estimated Source 2')
    axs[0][2].title.set_text('Mixture (Blue) vs Sum of estimated sources')

    for i,j in enumerate([0,7,2,23]):
        axs[i][0].plot(extractedSpikes[j])
        axs[i][1].plot(extractedNoises[j])
        axs[i][2].plot(mixes[j])
        axs[i][2].plot(generatedMixes[j])

    plt.show()
    
    return (extractedSpikes, extractedNoises)

def interpretSeparation(extractedSpikesValidation, critic, vallabel, method="energy", test=False, testThreshold=0):
    if method == "energy":
        energy = []

        for extractedSpike in extractedSpikesValidation:
            energy.append(sum(np.power(extractedSpike, 2)))

        energy = np.array(energy)
        hist, edges = np.histogram(energy, bins = 100)

        plt.plot(np.sort(energy))
        plt.show()
        plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black", align="edge")
        plt.show()

        thresholds =  [x / 20 for x in range(1,600,1)]

        precisions = []
        recalls = []
        f1s = []
        accs = []
        used = []

        for threshold in thresholds:
            try:
                res = []

                for elem in energy:
                    if elem > threshold:
                        res.append(1)
                    else:
                        res.append(0)

                truepos = 0
                falsepos = 0
                trueneg = 0
                falseneg = 0

                for i in range(0, len(vallabel)):
                    if vallabel[i] == 1:
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
                f1 = 2 * ((precision * recall)/(precision+recall))
                acc = (truepos + trueneg) / len(vallabel)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                accs.append(acc)
                used.append(threshold)
            except:
                continue
            
        _, axs = plt.subplots(1,2)
        axs[0].plot(used)
        axs[0].title.set_text("Used treshold")
        axs[1].plot(accs)
        axs[1].title.set_text("Accuracy")
        plt.show()
        _, axs = plt.subplots(1,2)
        axs[0].plot(used)
        axs[0].title.set_text("Used treshold")
        axs[1].plot(precisions)
        axs[1].title.set_text("Precision")
        plt.show()
        _, axs = plt.subplots(1,2)
        axs[0].plot(used)
        axs[0].title.set_text("Used treshold")
        axs[1].plot(recalls)
        axs[1].title.set_text("Recall")
        plt.show()
        _, axs = plt.subplots(1,2)
        axs[0].plot(used)
        axs[0].title.set_text("Used treshold")
        axs[1].plot(f1s)
        axs[1].title.set_text("F1")
        plt.show()

        f1max = f1s[0]
        maxI = 0

        for i,f in enumerate(f1s):
            if f > f1max:
                maxI = i
                f1max = f
        print("Threshold for best F1: {}".format(used[maxI]))
        res = []
        threshold = used[maxI]

        if test:
            threshold = testThreshold

        print("Threshold: {}".format(threshold))

        for elem in energy:
            if elem > threshold:
                res.append(1)
            else:
                res.append(0)

        truepos = 0
        falsepos = 0
        trueneg = 0
        falseneg = 0

        for i in range(0, len(vallabel)):
            if vallabel[i] == 1:
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
        f1 = 2 * ((precision * recall)/(precision+recall))
        acc = (truepos + trueneg) / len(vallabel)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("Accuracy: {}".format(acc))
    
    if method == "critic":
        extractedSpikesValidation = torch.from_numpy(extractedSpikesValidation)

        extractedSpikesValidation = torch.fft.rfft(extractedSpikesValidation, dim=1)
        extractedSpikesValidation = torch.stack([torch.cat((x.real,x.imag),0) for x in extractedSpikesValidation])
        criticScores = critic.forward(extractedSpikesValidation.to(device)).cpu().detach().numpy()
        criticScores = [x for [x] in criticScores]

        hist, edges = np.histogram(criticScores, bins = 30)

        plt.plot(np.sort(criticScores))
        plt.show()
        plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black", align="edge")
        plt.show()

        thresholds =  [x / 10 for x in range(-600,-300,1)]

        precisions = []
        recalls = []
        f1s = []
        accs = []
        used = []

        for threshold in thresholds:
            try:
                res = []

                for elem in criticScores:
                    if elem > threshold:
                        res.append(1)
                    else:
                        res.append(0)

                truepos = 0
                falsepos = 0
                trueneg = 0
                falseneg = 0

                for i in range(0, len(vallabel)):
                    if vallabel[i] == 1:
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
                f1 = 2 * ((precision * recall)/(precision+recall))
                acc = (truepos + trueneg) / len(vallabel)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                accs.append(acc)
                used.append(threshold)
            except:
                continue
            
        _, axs = plt.subplots(1,2)
        axs[0].plot(used)
        axs[0].title.set_text("Used treshold")
        axs[1].plot(accs)
        axs[1].title.set_text("Accuracy")
        plt.show()
        _, axs = plt.subplots(1,2)
        axs[0].plot(used)
        axs[0].title.set_text("Used treshold")
        axs[1].plot(precisions)
        axs[1].title.set_text("Precision")
        plt.show()
        _, axs = plt.subplots(1,2)
        axs[0].plot(used)
        axs[0].title.set_text("Used treshold")
        axs[1].plot(recalls)
        axs[1].title.set_text("Recall")
        plt.show()
        _, axs = plt.subplots(1,2)
        axs[0].plot(used)
        axs[0].title.set_text("Used treshold")
        axs[1].plot(f1s)
        axs[1].title.set_text("F1")
        plt.show()

        f1max = f1s[0]
        maxI = 0

        for i,f in enumerate(f1s):
            if f > f1max:
                maxI = i
                f1max = f
        print("Threshold for best F1: {}".format(used[maxI]))
        res = []
        threshold = used[maxI]

        if test:
            threshold = testThreshold

        print("Threshold: {}".format(threshold))

        for elem in criticScores:
            if elem > threshold:
                res.append(1)
            else:
                res.append(0)

        truepos = 0
        falsepos = 0
        trueneg = 0
        falseneg = 0

        for i in range(0, len(vallabel)):
            if vallabel[i] == 1:
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
        f1 = 2 * ((precision * recall)/(precision+recall))
        acc = (truepos + trueneg) / len(vallabel)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("Accuracy: {}".format(acc))