import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


#define the models
class DebugLayer(nn.Module):
    def __init__(self, log):
        super(DebugLayer, self).__init__()
        self.log = log
    
    def forward(self, x):
        print(self.log)
        print(x.shape)
        return x
    
class Conv2dAdapter(nn.Module):
    def __init__(self, outputSize, isIn):
        super(Conv2dAdapter, self).__init__()
        self.outputSize = outputSize
        self.isIn = isIn
    
    def forward(self, x):
        if self.isIn:
            s = int(np.sqrt(self.outputSize))
            out = x.view(-1, 1, s, s)
        else:
            out = x.view(-1, self.outputSize)
        return out

class netG_images(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(netG_images, self).__init__()
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
            Conv2dAdapter(self.outputSize, True),
            #DebugLayer("conv adapter in"),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,
                     padding=2),
            #DebugLayer("gen conv")
            Conv2dAdapter(self.outputSize, False),
            #DebugLayer("conv adapter out"),
            nn.Softplus(),
            #DebugLayer("gen third sf")
        )
        
    def forward(self, x):
        return self.mainModule(x.view(-1,self.inputSize))


class netD_images(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(netD_images, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize

        self.mainModule = nn.Sequential(
            #DebugLayer("disc input"),
            nn.Linear(self.inputSize, self.hiddenSize, bias=True),
            #DebugLayer("disc first lin"),
            nn.Tanh(),
            #DebugLayer("disc tanh"),
            nn.Linear(self.hiddenSize, 1, bias=True),
            #DebugLayer("disc second lin"),
        )
        
    def forward(self, x):
        return self.mainModule(x.view(-1,self.inputSize))

def adversarial_trainer( 
    train_loader, 
    noise_loader,
    generator, 
    discriminator, 
    epochs=5,
    learningRate=0.001,
    criterion=nn.BCELoss(),
    clampLower=-0.01,
    clampHigher=0.01,
    Diters=5,
    Giters=1
):
    inputSize, outputSize = generator.inputSize, generator.outputSize
    
    optimizerD = optim.RMSprop(discriminator.parameters(), lr=learningRate)
    optimizerG = optim.RMSprop(generator.parameters(), lr=learningRate)
    
    my_dpi = 96
    plt.figure(figsize=(1200/my_dpi, 600/my_dpi), dpi=my_dpi)
    true, false = 1, 0
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    
    dRealErr = []
    dFakeErr = []
    gErr = []
    
    for epoch in range(epochs):
        for batch, ([ft], [noise]) in enumerate(zip(train_loader, noise_loader)):
            ft = ft.squeeze().to(device)
            noise = noise.to(device)

            for disc_ep in range(Diters):
                for p in discriminator.parameters():
                    p.data.clamp_(clampLower, 
                                  clampHigher)


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

                err_D = err_D_real - err_D_fake
                optimizerD.step()

            # generator gradient
            for gEpoch in range(Giters):
                generator.zero_grad()
                out_h_data = discriminator.forward(ft)    
                out_h_g = discriminator.forward(out_g) 
                #err_G = ((out_h_data.mean(0) - out_h_g.mean(0))**2).sum()
                err_G = torch.mean(out_h_g)
                err_G.backward()

                optimizerG.step()

        # show the current generated output
        dRealErr
        dFakeErr
        gErr
        if (epoch + 1) % 10 == 0:
            showImageGrid(out_g.cpu().detach().numpy(), 2, 2, 28, 28, [4,4])
            print("out_d {}".format(out_d.mean()))
            print("out_d_g {}".format(out_d_g.mean()))
            print("err_G {}".format(err_G.mean()))
            print("Epoch {}".format(epoch))