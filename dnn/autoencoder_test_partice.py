# @Time:2020/8/7 19:33
# @Author:liuAmon
# @e-mail:utopfish@163.com
# @File:autoencoder_test_partice.py
__author__ = "liuAmon"

import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, dim, theta):
        super(Autoencoder, self).__init__()
        self.dim = dim
        # self.drop_out = nn.Dropout(p=0.1)
        self.encoder = nn.Sequential(
            nn.Linear(dim + theta * 0, dim + theta * 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim + theta * 1, dim + theta * 0),
        )

    def forward(self, x):
        x = x.view(-1, self.dim)

        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, self.dim)

        return out

class StockedAutoencoder(nn.Module):
    def __init__(self, dim, theta):
        super(StockedAutoencoder, self).__init__()
        self.dim = dim
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

        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, self.dim)

        return out

class ResAutoencoder(nn.Module):
    def __init__(self, dim, theta):
        super(ResAutoencoder, self).__init__()
        self.dim = dim


        self.encoder = nn.Sequential(
            nn.Linear(dim + theta * 0, dim + theta * 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim + theta * 1, dim + theta * 0),
        )

    def forward(self, x):
        x = x.view(-1, self.dim)
        # x_missed = self.drop_out(x)

        z = self.encoder(x)
        out = self.decoder(z)

        out = out + x
        out = out.view(-1, self.dim)

        return out

class StockedResAutoencoder(nn.Module):
    def __init__(self, dim, theta):
        super(StockedResAutoencoder, self).__init__()
        self.dim = dim
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
        # x_missed = self.drop_out(x)

        z = self.encoder(x)
        out = self.decoder(z)

        out = out + x
        out = out.view(-1, self.dim)

        return out
