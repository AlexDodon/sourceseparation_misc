
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch 
from torch.autograd import Variable

from torchgan.layers import MinibatchDiscrimination1d

import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _DebugLayer(nn.Module):
    def __init__(self, log):
        super(_DebugLayer, self).__init__()
        self.log = log
    
    def forward(self, x):
        print(self.log)
        print(x.shape)
        return x

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
                #DebugLayer("gen input"),
                nn.Linear(self.inputSize, self.hiddenSize, bias=True),
                #DebugLayer("gen first lin"),
                nn.Softplus(),
                #DebugLayer("gen first sf"),
                nn.Linear(self.hiddenSize, self.outputSize, bias=True),
                #DebugLayer("gen second lin"),
                nn.Softplus(),
                #DebugLayer("gen second sf"),
        )
        
         
    def forward(self, x):
        return self.mainModule(x.view(-1,self.inputSize))

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

    def __init__(self, inputSize,hiddenSize):
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
      
    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()     



class Crit(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Crit,self).__init__()
        self.input=input_size
        self.hidden=hidden_size
        self.output=output_size
        self.mainModule=nn.Sequential(
            nn.Linear(self.input,self.hidden,bias=True),
            MinibatchDiscrimination1d(self.hidden,self.hidden,16),
            nn.Linear(self.hidden+self.hidden,1,bias=True),
        )
        
        
    def forward(self,x):
        
        return self.mainModule(x)


def MyCrit(input_size, hidden_size,output_size):
    return Crit(input_size,hidden_size,output_size).to(device)


def Generator(inputSize, hiddenSize, outputSize):
    return _netG(inputSize, hiddenSize, outputSize).to(device)


def Critic(inputSize,hiddenSize):
    return _netC(inputSize,hiddenSize).to(device)

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




unrolled_steps=10
def adversarial_trainer( 
    train_loader, 
    generator, 
    critic, 
    device = device,
    epochs=5,
    learningRate=1e-4,
    Citers=5,
    Giters=1,
    noiseDim = 100,
    batchSize = 64,
    gpLambda = 10, 
    printEpochs=10,
    examples=4,
    noise=None
):
    #optimizerC = optim.Adam(critic.parameters(), lr=learningRate, betas=(0.5, 0.9))
    #optimizerG = optim.Adam(generator.parameters(), lr=learningRate, betas=(0.5, 0.9))
    optimizerC = optim.RMSprop(critic.parameters(), lr=learningRate)
    optimizerG = optim.RMSprop(generator.parameters(), lr=learningRate)
    
    my_dpi = 96
    
    criticLoss = []
    
    true, false = 1, 0
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    
    dRealErr = []
    dFakeErr = []
    gErr = []

    for epoch in range(epochs):
        for [real] in train_loader:
            real = real.squeeze().to(device)

            for _ in range(Citers):
               noise = torch.rand(real.shape[0], 1, noiseDim).to(device)
               critic.zero_grad()
               out_d=critic.forward(real)
               err_D_real=out_d.mean()
               err_D_real.backward(one)

               out_g=generator.forward(noise)
               out_d_g=critic.forward(Variable(out_g.data))
               err_D_fake=out_d_g.mean()
               err_D_fake.backward(mone)
               err_D=err_D_real-err_D_fake
               optimizerC.step()


               for p in critic.parameters():
                    p.data.clamp_(-0.01,0.01)
            
            #for _ in range(Citers):
            #
            #    noise = torch.rand(real.shape[0], 1, noiseDim).to(device)
#
            #    fake = generator.forward(noise)
#
#
            #   
            #
            #    critic_real = critic.forward(real).reshape(-1)
            #   
            #    critic_fake = critic.forward(fake).reshape(-1)
            #    
            #    #print(noise.size())
            #    #print(fake.size())
            #    #print(critic_real.size())
            #    #print(critic_fake.size())
            #    #print("asta")
            #    # gp = _gradient_penalty(critic, real, fake)
#
            #    critic_loss = (
            #        torch.mean(critic_fake)     # Tries to minimize critic_fake
            #        -torch.mean(critic_real)    # Tries to maximize critic_real
            #    #    + gpLambda * gp             # Tries to minimize gradient penalty
            #    )
#
            #    critic.zero_grad()
            #    critic_loss.backward(retain_graph=True)
            #    optimizerC.step()
#
            #    for p in critic.parameters():
            #        p.data.clamp_(-0.01,0.01)

            #for _ in range(Giters):
            #    noise = torch.rand(batchSize, 1, noiseDim).to(device)
            #   
            #    fake = generator.forward(noise)
            #    critic_fake = critic.forward(fake).reshape(-1)
            #    
            #    generator_loss = -torch.mean(critic_fake) # Tries to maximize critic_fake
#
            #    generator.zero_grad()
            #    generator_loss.backward(retain_graph=True)
            #    optimizerG.step()
            # 

            for _ in range(Giters):
                generator.zero_grad()
                out_h_data = critic.forward(real)    
                out_h_g = critic.forward(out_g) 
                #err_G = ((out_h_data.mean(0) - out_h_g.mean(0))**2).sum()
                err_G = torch.mean(out_h_g)
                err_G.backward()

                optimizerG.step()

            #criticLoss.append(critic_loss.cpu().detach().numpy())
            dRealErr.append(err_D_real.cpu().detach().numpy())
            dFakeErr.append(err_D_fake.cpu().detach().numpy())
            gErr.append(err_G.mean().cpu().detach().numpy())


        if (epoch + 1) % printEpochs == 0:
            fig, axs = plt.subplots(3)
            axs[0].plot(dRealErr)
            axs[1].plot(dFakeErr)
            axs[2].plot(gErr)
            
            print("\nEpoch {}".format(epoch))
            plt.show()
            
            print("\nGenerated example:")
            
            
            fig, axs = plt.subplots(4,2)
            
            axs[0][0].plot(generator.forward(noise).cpu().detach().numpy()[0])
            axs[0][1].plot(generator.forward(noise).cpu().detach().numpy()[1])
            axs[1][0].plot(generator.forward(noise).cpu().detach().numpy()[2])
            axs[1][1].plot(generator.forward(noise).cpu().detach().numpy()[3])
            axs[2][0].plot(generator.forward(noise).cpu().detach().numpy()[4])
            axs[2][1].plot(generator.forward(noise).cpu().detach().numpy()[5])
            axs[3][0].plot(generator.forward(noise).cpu().detach().numpy()[6])
            axs[3][1].plot(generator.forward(noise).cpu().detach().numpy()[7])
            plt.show()

            print("err_d {}".format(out_d.mean()))
            print("err_d_g {}".format(out_d_g.mean()))
            print("err_G {}".format(err_G.mean()))

if __name__ =="__main__":
    print("No main module functionality.")