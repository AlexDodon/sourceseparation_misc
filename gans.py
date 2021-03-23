
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch 
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DebugLayer(nn.Module):
    def __init__(self, log):
        super(DebugLayer, self).__init__()
        self.log = log
    
    def forward(self, x):
        print(self.log)
        print(x.shape)
        return x
    
class netG(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(netG, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        
        self.mainModule = nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize, bias=True),
            nn.Softplus(),
            nn.Linear(self.hiddenSize, self.outputSize, bias=True),
            nn.Softplus(),
        )
        
    def forward(self, x):
        return self.mainModule(x.view(-1,self.inputSize))


class netD(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(netD, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize

        self.mainModule = nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize, bias=True),
            nn.Tanh(),
            nn.Linear(self.hiddenSize, 1, bias=True),
        )
        
    def forward(self, x):
        return self.mainModule(x.view(-1,self.inputSize))

def Generator(inputSize, hiddenSize, outputSize):
    return netG(inputSize, hiddenSize, outputSize).to(device)


def Critic(inputSize, hiddenSize):
    return netD(inputSize, hiddenSize).to(device)

def adversarial_trainer( 
    train_loader, 
    noise_loader,
    generator, 
    discriminator, 
    device = device,
    epochs=5,
    learningRate=0.001,
    criterion=nn.BCELoss(),
    clampLower=-0.01,
    clampHigher=0.01,
    Diters=5,
    Giters=1,
    printEpochs=10
):
    optimizerD = optim.RMSprop(discriminator.parameters(), lr=learningRate)
    optimizerG = optim.RMSprop(generator.parameters(), lr=learningRate)
    
    my_dpi = 96
    plt.figure(figsize=(600/my_dpi, 300/my_dpi), dpi=my_dpi)
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    
    dRealErr = []
    dFakeErr = []
    gErr = []
    
    for epoch in range(epochs):
        for _, ([ft], [noise]) in enumerate(zip(train_loader, noise_loader)):
            ft = ft.squeeze().to(device)
            noise = noise.to(device)

            for _ in range(Diters):
                for p in discriminator.parameters():
                    p.data.clamp_(clampLower, clampHigher)


                # discriminator gradient with real data
                discriminator.zero_grad()
                out_d = discriminator.forward(ft)

                err_D_real = out_d.mean()
                err_D_real.backward(one)

                # discriminator gradient with generated data
                
                out_g = generator.forward(noise)
                out_d_g = discriminator.forward(Variable(out_g.data))
                err_D_fake = out_d_g.mean()
                err_D_fake.backward(mone)

                optimizerD.step()

            # generator gradient
            for gEpoch in range(Giters):
                generator.zero_grad()
                out_g = generator.forward(noise)
                out_h_g = discriminator.forward(out_g) 
                err_G = torch.mean(out_h_g)
                err_G.backward()

                optimizerG.step()
                
            dRealErr.append(err_D_real.cpu().detach().numpy())
            dFakeErr.append(err_D_fake.cpu().detach().numpy())
            gErr.append(err_G.mean().cpu().detach().numpy())

        # show the current generated output
        if (epoch + 1) % printEpochs == 0:
            plt.plot(dRealErr)
            plt.title("Critic error on real data")
            plt.show()
            plt.plot(dFakeErr)
            plt.title("Critic error on generated data")
            plt.show()
            plt.plot(gErr)
            plt.title("Generator error")
            plt.show()
            
            print("\nEpoch {}".format(epoch))
            
            print("\nGenerated example:")
            
            
            _, axs = plt.subplots(2,2)
            
            axs[0][0].plot(generator.forward(noise).cpu().detach().numpy()[0])
            axs[0][1].plot(generator.forward(noise).cpu().detach().numpy()[1])
            axs[1][0].plot(generator.forward(noise).cpu().detach().numpy()[2])
            axs[1][1].plot(generator.forward(noise).cpu().detach().numpy()[3])
            plt.show()

            print("err_d {}".format(out_d.mean()))
            print("err_d_g {}".format(out_d_g.mean()))
            print("err_G {}".format(err_G.mean()))

if __name__ =="__main__":
    print("No main module functionality.")