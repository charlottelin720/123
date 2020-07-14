#!/usr/bin/env python
# coding: utf-8

# In[136]:


import numpy as np
import torch, os, glob
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(0)
device = torch.device('cpu')
# torch.cuda.set_device(0)

# global variable




# In[137]:


# cutting pictures into assigned size and save them in files
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalization 

BATCH_SIZE = 1024
EPOCH = 300

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),])
# must be file : data
pic_set = datasets.ImageFolder('.\data', transform)
pics_loader = torch.utils.data.DataLoader(dataset=pic_set, batch_size=BATCH_SIZE, shuffle=True)


# In[138]:


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # encoder的部分
        self.conv_encode = nn.Sequential( 
            # (RGB, output_nodes, kernel_size=3, stride=2, padding=1)
            nn.Conv2d(3, 64,  kernel_size=3, stride=2, padding=1), #(64, 64, 3) -> (32, 32, 64)
            nn.BatchNorm2d(64),   #2D Normalization   (num_features, eps=1e-05, momentum=0.1, affine=True)
            nn.ReLU(),
            
            nn.Conv2d(64, 64,  kernel_size=3, stride=2, padding=1), #(32, 32, 64) -> (16, 16, 64)
            nn.BatchNorm2d(64),   #2D Normalization   (num_features, eps=1e-05, momentum=0.1, affine=True)
            nn.ReLU(),
            
            # (input_nodes, output_nodes, kernel_size=3, stride=2, padding=1)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), #(16, 16, 64) -> (8, 8, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # (input_nodes, output_nodes, kernel_size=3, stride=2, padding=1)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), #(8, 8, 128) -> (4, 4, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(),)
        
        # (in_features, out_features)
        self.generate_mean = nn.Linear(4*4*256, 128) # to generate mean
        self.generate_var = nn.Linear(4*4*256, 128) # to generate log variance
    
        self.fc = nn.Linear(128, 4*4*256) 
        # decoder的部分
        self.conv_decode = nn.Sequential( 
            # (input_nodes, output_nodes, kernel_size=3, stride = 2, padding = 1, output_padding = 1)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride = 2, padding = 1, output_padding = 1), #(4, 4, 256) -> (8, 8, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # (input_nodes, output_nodes, kernel_size=3, stride = 2, padding = 1, output_padding = 1)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride = 2, padding = 1, output_padding = 1), #(8, 8, 128) -> (16, 16, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # (input_nodes, output_nodes, kernel_size=3, stride = 2, padding = 1, output_padding = 1)
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride = 2, padding = 1, output_padding = 1), #(16, 16, 64) -> (32, 32, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # (input_nodes, output_nodes, kernel_size=3, stride = 2, padding = 1, output_padding = 1)
            nn.ConvTranspose2d(64, 3,  kernel_size=3, stride = 2, padding = 1, output_padding = 1), #(32, 32, 64) -> (64, 64, 3)
            nn.BatchNorm2d(3),
            nn.Sigmoid(),)


    # model(batch_X, batch_X.size(0))
    def encoder(self, x):
        # encode and reshape
        h = self.conv_encode(x)
        h = h.view(h.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
        # generate mean and std diviation
        mu = self.generate_mean(h)
        var = self.generate_var(h)
        
        return mu, var

    def decoder(self, x, batch_size):
        h = self.fc(x)
        h = h.view(batch_size, 256, 4, 4) # reshape to image
        output = self.conv_decode(h)
        return output
    
    def forward(self, x, batch_size):
        mu, var = self.encoder(x)
        z = self.reparameterize(mu, var)
        x_reconst = self.decoder(z, batch_size)
        
        return x_reconst, mu, var
    
    
        
    def reparameterize(self, mu, var):
        std = torch.exp(var/2)
        epslion = torch.randn_like(std)
        z = mu + std * epslion
        
        return z
    
    def vae_loss(self, recon_x, x, mu, logvar):
 
        recon_loss = F.binary_cross_entropy(x_reconst, batch_X, size_average=False)

        kl_Loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ELBO = recon_loss + kl_Loss
    
        return ELBO
                                                                        
                                            
    def vae_loss_0(self, recon_x, x, mu, logvar):
 
        recon_loss = F.binary_cross_entropy(x_reconst, batch_X, size_average=False)

        kl_Loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ELBO = recon_loss + (0 * kl_Loss)
    
        return ELBO
                                                                         
    
    def vae_loss_100(self, recon_x, x, mu, logvar):
 
        recon_loss = F.binary_cross_entropy(x_reconst, batch_X, size_average=False)

        kl_Loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ELBO = recon_loss + (100 * kl_Loss)
    
        return ELBO
                                                               
    

    def plot_imgs(imgs, title):
        fig = plt.gcf()
        fig.set_size_inches((12, 5))
        fig.suptitle(title, fontsize=20, color='b')
        for i in range(10):
            ax = plt.subplot(2, 5, i+1)
            ax.imshow(imgs[i].permute(1, 2, 0))

    def show_image(self, imgs, title):
        x = np.transpose(torchvision.utils.make_grid(imgs[:50],10,5), (1, 2, 0))
        print(title)
        plt.imshow(x)
        
        
        

model = VAE()
model = model.to(device)

# In[139]:


# training
history_loss = []
start_time = time.time()

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(EPOCH):
    model.train()
    for idx, (batch_X, batch_y) in enumerate(pics_loader):
        batch_X = batch_X.to(device)
        x_reconst, mu, var = model(batch_X, batch_X.size(0))
        
        ELBO = model.vae_loss(x_reconst, batch_X, mu, var)

        
        optimizer.zero_grad()
        ELBO.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), 'VAE_model.pth')
    print('Epoch:', epoch, '--> loss:', ELBO.item())
    history_loss.append(ELBO.item())
    
    #print imgs, wipe out memory every 10 epoch
    if(epoch % 10 == 0):
        with torch.no_grad():
            model.eval()
            #????????????
            sample = torch.randn(BATCH_SIZE, 128).to(device)
            imgs = model.decoder(sample, batch_size=BATCH_SIZE).cpu()
            model.show_image(imgs, 'Epoch:'+ str(epoch))
#         plt.savefig('training(epoch:' + str(epoch) + ').png')
        
        



# In[140]:


# Plot the learning curve of the negative evidence lower bound (ELBO) of log likelihood of training images.
plt.plot(history_loss)
plt.title('learning curve of the negative evidence lower bound (ELBO)', color='b')
plt.xlabel('Epochs')
# plt.ylabel('Loss')


# In[141]:


# 依照 idx 選幾個印出
fig = plt.gcf()
fig.set_size_inches((12, 10))
fig.suptitle('Show some examples reconstructed by your model.', fontsize=20, color='b')
idx = 0
pos_idx = 0
for _ in range(20):
    if(pos_idx == 20):
        break
    ax = plt.subplot(5, 4, pos_idx+1)
    ax.imshow(batch_X[idx].permute(1, 2, 0))
    
    ax = plt.subplot(5, 4, pos_idx+2)
    ax.imshow(x_reconst[idx].detach().numpy().transpose(1, 2, 0))
    pos_idx += 2
    idx += 1


# In[143]:


# Show the synthesized images based on the interpolation of two latent codes z between two real samples.

def interpolation(lambda1, model, img1, img2):
    
    with torch.no_grad():
    
        # latent vector of first image
#         img1 = img1.to(device)
#         img_1 = img1.view(64, 3, 3, 3)
#         latent_1, _ = model(img1, img1.size(0))
        latent_1, _ = model.encoder(img1)

        # latent vector of second image
#         img2 = img2.to(device)
#         img_2 = img2.view(64, 3, 3, 3)
#         latent_2, _ = model(img2, img2.size(0))
        latent_2, _ = model.encoder(img2)

        # interpolation of the two latent vectors
        inter_latent = lambda1* latent_1 + (1- lambda1) * latent_2

        # reconstruct interpolated image
        inter_image = model.decoder(inter_latent, 1)
        inter_image = inter_image.cpu()

        return inter_image
    
# interpolation lambdas
lambda_range=np.linspace(0,1,10)

fig, axs = plt.subplots(2,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()

for ind,l in enumerate(lambda_range):
    inter_image=interpolation(float(l), model, pic_set[0][0].unsqueeze(0), pic_set[10][0].unsqueeze(0))
   
    inter_image = inter_image.clamp(0, 1)
    
    image = inter_image.numpy()
   
    axs[ind].imshow(image[0,:,:,:].transpose(1,2,0))
    axs[ind].set_title('lambda_val='+str(round(l,1)))
plt.show() 


# In[132]:


# KL = 0
# KL = 0
# KL = 0
# KL = 0
history_loss = []
start_time = time.time()

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(EPOCH):
    model.train()
    for idx, (batch_X, batch_y) in enumerate(pics_loader):
        batch_X = batch_X.to(device)
        x_reconst, mu, var = model(batch_X, batch_X.size(0))
        
        ELBO = model.vae_loss_0(x_reconst, batch_X, mu, var)

        
        optimizer.zero_grad()
        ELBO.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), 'VAE_model.pth')
    print('Epoch:', epoch, '--> loss:', ELBO.item())
    history_loss.append(ELBO.item())
    
    #print imgs, wipe out memory every 10 epoch
    if(epoch % 10 == 0):
        with torch.no_grad():
            model.eval()
            #????????????
            sample = torch.randn(BATCH_SIZE, 128).to(device)
            imgs = model.decoder(sample, batch_size=BATCH_SIZE).cpu()
            model.show_image(imgs, 'Epoch:'+ str(epoch))
#         plt.savefig('training(epoch:' + str(epoch) + ').png')
        
        


# In[ ]:


# Plot the learning curve of the negative evidence lower bound (ELBO) of log likelihood of training images.
plt.plot(history_loss)
plt.title('learning curve of the negative evidence lower bound (ELBO)', color='b')
plt.xlabel('Epochs')
# plt.ylabel('Loss')


# In[ ]:


# 依照 idx 選幾個印出
fig = plt.gcf()
fig.set_size_inches((12, 10))
fig.suptitle('Show some examples reconstructed by your model.', fontsize=20, color='b')
idx = 0
pos = 0
for _ in range(20):
    if(pos == 20):
        break
    ax = plt.subplot(5, 4, pos+1)
    ax.imshow(batch_X[idx].permute(1, 2, 0))
    
    ax = plt.subplot(5, 4, pos+2)
    ax.imshow(x_reconst[idx].detach().numpy().transpose(1, 2, 0))
    pos += 2
    idx += 1


# In[ ]:


# Show the synthesized images based on the interpolation of two latent codes z between two real samples.
#KL-0
# interpolation lambdas
lambda_range=np.linspace(0,1,10)

fig, axs = plt.subplots(2,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()

for ind,l in enumerate(lambda_range):
    inter_image=interpolation(float(l), model, pic_set[0][0].unsqueeze(0), pic_set[10][0].unsqueeze(0))
   
    inter_image = inter_image.clamp(0, 1)
    
    image = inter_image.numpy()
   
    axs[ind].imshow(image[0,:,:,:].transpose(1,2,0))
    axs[ind].set_title('lambda_val='+str(round(l,1)))
plt.show() 


# In[ ]:


# KL = 100
# KL = 100
# KL = 100
# KL = 100
history_loss = []
start_time = time.time()

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(EPOCH):
    model.train()
    for idx, (batch_X, batch_y) in enumerate(pics_loader):
        batch_X = batch_X.to(device)
        x_reconst, mu, var = model(batch_X, batch_X.size(0))
        
        ELBO = model.vae_loss_100(x_reconst, batch_X, mu, var)

        
        optimizer.zero_grad()
        ELBO.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), 'VAE_model.pth')
    print('Epoch:', epoch, '--> loss:', ELBO.item())
    history_loss.append(ELBO.item())
    
    #print imgs, wipe out memory every 10 epoch
    if(epoch % 10 == 0):
        with torch.no_grad():
            model.eval()
            #????????????
            sample = torch.randn(BATCH_SIZE, 128).to(device)
            imgs = model.decoder(sample, batch_size=BATCH_SIZE).cpu()
            model.show_image(imgs, 'Epoch:'+ str(epoch))
#         plt.savefig('training(epoch:' + str(epoch) + ').png')
        
        

# In[ ]:



plt.plot(history_loss)
plt.title('learning curve of the negative evidence lower bound (ELBO)', color='b')
plt.xlabel('Epochs')
# plt.ylabel('Loss')


# In[144]:


# 依照 idx 選幾個印出
fig = plt.gcf()
fig.set_size_inches((12, 10))
fig.suptitle('Show some examples reconstructed by your model.', fontsize=20, color='b')
idx = 0
pos_idx = 0
for _ in range(20):
    if(pos_idx == 20):
        break
    ax = plt.subplot(5, 4, pos_idx+1)
    ax.imshow(batch_X[idx].permute(1, 2, 0))
    
    ax = plt.subplot(5, 4, pos_idx+2)
    ax.imshow(x_reconst[idx].detach().numpy().transpose(1, 2, 0))
    pos_idx += 2
    idx += 1


# In[145]:


# Synthesized samples drawn from VAE
with torch.no_grad():
    model.eval()
    sample = torch.randn(BATCH_SIZE, 128).to(device)
    # decoder(x, batch_size):
    imgs = model.decoder(sample, batch_size=BATCH_SIZE).cpu()
    model.show_image(imgs, 'Synthesized samples drawn from VAE')


# In[ ]:


# Show the synthesized images based on the interpolation of two latent codes z between two real samples.

#KL-100
# interpolation lambdas
lambda_range=np.linspace(0,1,10)

fig, axs = plt.subplots(2,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()

for ind,l in enumerate(lambda_range):
    inter_image=interpolation(float(l), model, pic_set[0][0].unsqueeze(0), pic_set[10][0].unsqueeze(0))
   
    inter_image = inter_image.clamp(0, 1)
    
    image = inter_image.numpy()
   
    axs[ind].imshow(image[0,:,:,:].transpose(1,2,0))
    axs[ind].set_title('lambda_val='+str(round(l,1)))
plt.show() 

