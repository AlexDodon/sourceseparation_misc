
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch 
import torchgan
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['figure.figsize'] = [16, 8]

class _DebugLayer(nn.Module):
    def __init__(self, log):
        super(_DebugLayer, self).__init__()
        self.log = log
    
    def forward(self, x):
        print(self.log)
        print(x.shape)
        return x

class _FFTLayer(nn.Module):
    def __init__(self):
        super(_FFTLayer,self).__init__()

    def forward(self, x):
        fft = torch.fft.rfft(x, dim=1)
        return torch.stack([torch.cat((x.real,x.imag),0) for x in fft])

class _Conv1dAdapter(nn.Module):
    def __init__(self, outputSize, isIn):
        super(_Conv1dAdapter, self).__init__()
        self.outputSize = outputSize
        self.isIn = isIn
    
    def forward(self, x):
        if self.isIn:
            out = x.view(-1, 1, self.outputSize)
        else:
            out = x.view(-1, self.outputSize)
        return out
    
class _netG(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(_netG, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        
        self.mainModule = nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize, bias=True),
            nn.Softplus(),
            nn.Linear(self.hiddenSize, self.hiddenSize, bias=True),
            nn.Softplus(),
            nn.Linear(self.hiddenSize, self.outputSize, bias=True),
            # self._block(self.inputSize, 20),
            # self._block(20, 40),
            # self._block(40, 60),
            # self._block(60, self.outputSize),
        )
        
    def forward(self, x):
        return self.mainModule(x)

    def _block(self, inSize, outSize):
        return nn.Sequential(
            nn.Linear(inSize, outSize, bias=True),
            nn.Softplus(),
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=9,padding=4),
            nn.Softplus(),
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=9,padding=4),
            nn.Softplus(),
        )


class _netC(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(_netC, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize

        self.mainModule = nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize, bias=True),
            nn.Tanh(),
            nn.Linear(self.hiddenSize, 1, bias=True),
        )
        
    def forward(self, x):
        return self.mainModule(x)

def Generator(inputSize, hiddenSize, outputSize):
    return _netG(inputSize, hiddenSize, outputSize).to(device)


def Critic(inputSize, hiddenSize):
    return _netC(inputSize, hiddenSize).to(device)

def _gradient_penalty(critic, real, fake):
    batchSize, L = real.shape
    epsilon = torch.rand((batchSize, 1)).repeat(1,L).to(device)

    interpolation = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic.forward(interpolation)

    gradient = torch.autograd.grad(
        inputs=interpolation,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)

    gradient_norm = gradient.norm(2, dim=1)

    gradient_penalty = torch.mean((gradient_norm -1) ** 2)

    return gradient_penalty


def adversarial_trainer( 
    train_loader, 
    generator, 
    critic, 
    device = device,
    epochs=5,
    learningRate=1e-4,
    Citers=10,
    Giters=1,
    noiseDim = 100,
    batchSize = 64,
    gpLambda = 10, 
    printEpochs=10
):
    optimizerC = optim.Adam(critic.parameters(), lr=learningRate, betas=(0.0, 0.9))
    optimizerG = optim.Adam(generator.parameters(), lr=learningRate, betas=(0.0, 0.9))
    #optimizerC = optim.RMSprop(critic.parameters(), lr=learningRate)
    #optimizerG = optim.RMSprop(generator.parameters(), lr=learningRate)
    
    criticLoss = []
    
    for epoch in range(epochs):
        for [real] in train_loader:
            real = real.squeeze().to(device)

            for _ in range(Citers):
                noise = torch.rand(real.shape[0], 1, noiseDim).to(device)
                fake = generator.forward(noise).reshape(-1,critic.inputSize)

                critic_real = critic.forward(real).reshape(-1)
                critic_fake = critic.forward(fake).reshape(-1)

                gp = _gradient_penalty(critic, real, fake)

                critic_loss = (
                    torch.mean(critic_fake)     # Tries to minimize critic_fake
                    -torch.mean(critic_real)    # Tries to maximize critic_real
                    + gpLambda * gp             # Tries to minimize gradient penalty
                )

                critic.zero_grad()
                critic_loss.backward(retain_graph=True)
                optimizerC.step()

                # for p in critic.parameters():
                #     p.data.clamp_(-0.01,0.01)

            for _ in range(Giters):
                noise = torch.rand(batchSize, 1, noiseDim).to(device)
                fake = generator.forward(noise).reshape(-1,critic.inputSize)
                critic_fake = critic.forward(fake).reshape(-1)
                
                generator_loss = -torch.mean(critic_fake) # Tries to maximize critic_fake

                generator.zero_grad()
                generator_loss.backward()
                optimizerG.step()
            
            cLoss = critic_loss.cpu().detach().numpy()
            criticLoss.append(cLoss)

        if (epoch + 1) % printEpochs == 0:
            plt.plot(criticLoss)
            plt.plot(torch.zeros(len(criticLoss)).numpy())
            plt.title("Critic loss")
            plt.show()

            print("Critic loss {}".format(critic_loss))
            
            print("\nEpoch {}".format(epoch))
            
            print("\nGenerated example:")
            
            for i in [7,29,41,61]:
                plt.plot(torch.fft.irfft(fake[i][0:40] + 1j * fake[i][40:]).cpu().detach().numpy())
                plt.show()

if __name__ =="__main__":
    print("No main module functionality.")