import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.figsize'] = (8.0, 8.0)
BATCH_SIZE=512 #大概需要2G的显存
EPOCHS=20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)


def to_img(x):
    x = x.data.numpy()
    x = 0.5 * (x + 1)
    x = np.clip(x, 0, 1)
    x = x.reshape([-1, 28, 28])
    return x


def plot_reconstructions(model, save=True, name=None, conv=False, simple=False):
    """
    Plot 10 reconstructions from the baseline set. The top row is the original
    digits, the bottom is the decoder reconstruction.
    """
    # encode then decode
    data, _ = next(iter(test_loader))
    if not conv:
        data = data.view([-1, 784])
    data = Variable(data, volatile=True)
    true_imgs = data
    encoded_imgs = model.encoder(data)
    if simple:
        encoded_imgs = F.relu(encoded_imgs)
    decoded_imgs = model.decoder(encoded_imgs)

    true_imgs = to_img(true_imgs)
    decoded_imgs = to_img(decoded_imgs)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(true_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if save:
        plt.savefig('./plots/' + name + '.png', format='png', dpi=300)
    plt.show()
class Autoencoder(nn.Module):
    def __init__(self, dim,theta):
        super(Autoencoder, self).__init__()
        self.dim = dim

        self.drop_out = nn.Dropout(p=0.1)

        self.encoder = nn.Sequential(
            nn.Linear(dim + theta * 0, dim + theta * 1),
            nn.Tanh(),
            nn.Linear(dim + theta * 1, dim + theta * 2),
            nn.Tanh(),
            nn.Linear(dim + theta * 2, dim + theta * 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim + theta * 3, dim + theta * 2),
            nn.Tanh(),
            nn.Linear(dim + theta * 2, dim + theta * 1),
            nn.Tanh(),
            nn.Linear(dim + theta * 1, dim + theta * 0)
        )

    def forward(self, x):
        x = x.view(-1, self.dim)
        x_missed = self.drop_out(x)

        z = self.encoder(x_missed)
        out = self.decoder(z)

        out = out.view(-1, self.dim)

        return out,z


def l1_penalty(var):
    return torch.abs(var).sum()

n_features= 784
theta=10
model=Autoencoder(dim=n_features,theta=theta)
optimizer = optim.Adam(model.parameters())
def train(epoch, sparsity=False, l1_weight=1e-5):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data.view([-1, 784]))
        optimizer.zero_grad()

        # enforcing sparsity with l1 reg
        if sparsity:
            decoder_out,encoder_out = model(data)
            mse_loss = F.mse_loss(decoder_out, data)
            l1_reg = l1_weight * l1_penalty(encoder_out)
            loss = mse_loss + l1_reg
        else:
            output,_ = model(data)
            #loss = F.binary_cross_entropy_with_logits(output, data)
            loss = F.mse_loss(output, data)

        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))



num_epochs = 30

for epoch in range(1,  num_epochs + 1):
    train(epoch)

