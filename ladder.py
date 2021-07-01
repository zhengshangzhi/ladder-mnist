import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np

from dataset import MNISTTrainSet, MNISTTestSet


join = lambda l, u: torch.vstack((l, u))
labeled = lambda x: x[:batch_size]
unlabeled = lambda x: x[batch_size:]
split_lu = lambda x: (labeled(x), unlabeled(x))


def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean = batch.mean(0)
        var = batch.var(0)
    return (batch - mean) / torch.sqrt(var + 1e-10)


class Encoder(nn.Module):


    def __init__(self, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.shapes = zip(layer_sizes[:-1], layer_sizes[1:])
        self.L = len(layer_sizes) -1

        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(num_features, eps=1e-10, momentum=0.01, affine=False) for num_features in self.layer_sizes])

        self.W = nn.ParameterList([Parameter(torch.randn(s) / math.sqrt(s[0])) for s in self.shapes])
        self.gamma = nn.ParameterList([Parameter(torch.ones(layer_sizes[l+1])) for l in range(self.L)])
        self.beta = nn.ParameterList([Parameter(torch.zeros(layer_sizes[l+1])) for l in range(self.L)])

    def forward(self, x, noise):
        h = x + torch.randn(x.shape).to(device) * noise
        d = {}
        d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
        for l in range(1, self.L+1):
            d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
            z_pre = torch.matmul(h, self.W[l-1])
            z_pre_l, z_pre_u = split_lu(z_pre)
        
            m = z_pre_u.mean(0)
            v = z_pre_u.var(0)

            if self.training:
                if noise > 0:
                    z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
                    z += torch.randn(z_pre.shape).to(device) * noise
                else:
                    z = join(self.batchnorm[l](z_pre_l), batch_normalization(z_pre_u, m, v))
            else:
                z = self.batchnorm[l](z_pre)
            
            if l == L:
                h = F.softmax(self.gamma[l-1] * (z + self.beta[l-1]))
            else:
                h = F.relu(z + self.beta[l-1])
            d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
            d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v
        d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
        return h, d


class G_gauss(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.a1 = Parameter(torch.zeros(size))
        self.a2 = Parameter(torch.ones(size))
        self.a3 = Parameter(torch.zeros(size))
        self.a4 = Parameter(torch.zeros(size))
        self.a5 = Parameter(torch.zeros(size))
        self.a6 = Parameter(torch.zeros(size))
        self.a7 = Parameter(torch.ones(size))
        self.a8 = Parameter(torch.zeros(size))
        self.a9 = Parameter(torch.zeros(size))
        self.a10 = Parameter(torch.zeros(size))

    def forward(self, z_c, u):
        mu = self.a1 * F.sigmoid(self.a2 * u + self.a3) + self.a4 * u + self.a5
        v = self.a6 * F.sigmoid(self.a7 * u + self.a8) + self.a9 * u +self.a10

        z_est = (z_c - mu) * v + mu
        return z_est


class Decoder(nn.Module):

    def __init__(self, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.shapes = zip(layer_sizes[:-1], layer_sizes[1:])
        self.L = len(layer_sizes) -1

        self.V = nn.ParameterList([Parameter(torch.randn(s[::-1]) / math.sqrt(s[-1])) for s in self.shapes])
        self.g_gauss = nn.ModuleList([G_gauss(s).to(device) for s in self.layer_sizes])
    
    def forward(self, y_c, clean, corr):
        z_est = {}
        z_est_bn = {}
        for l in range(self.L, -1, -1):
            z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
            m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
            if l == self.L:
                u = unlabeled(y_c)
            else:
                u = torch.matmul(z_est[l+1], self.V[l])
            u = batch_normalization(u)
            z_est[l] = self.g_gauss[l](z_c, u)
            z_est_bn[l] = (z_est[l] - m) / v
        return z_est_bn


class Ladder(nn.Module):

    def __init__(self, layer_sizes, denoising_cost):
        super().__init__()
        self.denoising_cost = denoising_cost
        self.L = len(layer_sizes) - 1
        self.encoder = Encoder(layer_sizes)
        self.decoder = Decoder(layer_sizes)

    def forward(self, x, noise):
        if self.training:
            self.y_c, corr = self.encoder(x, noise)
            y, clean = self.encoder(x, 0.0)
            self.z = clean['unlabeled']['z']
            self.z_est_bn = self.decoder(self.y_c, clean, corr)
        else:
            y, clean = self.encoder(x, 0.0)
        return y

    def calc_loss(self, labels):
        mseloss = nn.MSELoss().to(device)
        d_cost = [mseloss(self.z_est_bn[l], self.z[l]) * self.denoising_cost[l] for l in range(self.L + 1)]
        u_cost = add_n(d_cost) * u_ratio
        y_l = labeled(self.y_c)
        l_cost = - (one_hot(labels) * torch.log(y_l)).sum(1).mean() * l_ratio
        loss = u_cost + l_cost
        return loss, l_cost, u_cost


def add_n(n_list):
    sum = torch.tensor(0.0).to(device)
    for n in n_list:
        sum += n
    return sum


def one_hot(x):
    y = torch.zeros(len(x), 10).to(device)
    for i in range(len(x)):
        y[i, x[i]].fill_(1.0)
    return y


def fit(model, trainloader, EPOCHES, LR):
    noise = 0.3
    decay_epoch = 15
    init = True
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHES):
        if epoch > decay_epoch:
            ratio = (EPOCHES - epoch) / (EPOCHES - decay_epoch)
            lr = LR * ratio
            adjust_lr(optim, lr)
        for x, labels in trainloader:
            model.train()
            x = x.to(device)
            labels = labels.to(device)
            y = model(x, noise)
            loss, l_cost, u_cost = model.calc_loss(labels)
            loss.backward()
            grad = model.encoder.W[1].grad.mean().cpu().numpy()
            optim.step()
            optim.zero_grad()
            if init:
                print(f'epoch: init, loss: {loss}, l_cost: {l_cost}, u_cost: {u_cost}, accuarcy: {evaluate(ladder, testimages, testlabels)}')
                init = False
        print(f'epoch: {epoch}, loss: {loss}, l_cost: {l_cost}, u_cost: {u_cost}, accuarcy: {evaluate(ladder, testimages, testlabels)}, grad: {grad}')


def adjust_lr(optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr

            
def evaluate(model, testdata, testlabels):
    model.eval()
    with torch.no_grad():
        testdata = testdata.to(device)
        testlabels = testlabels.to(device)
        y = model(testdata, 0.0)
        y = y.argmax(1)
        accuracy = ((y == testlabels).sum() / len(y) * 100)
    return accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_ratio = 1
    l_ratio = 1
    layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
    denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]
    L = len(layer_sizes) - 1
    batch_size = 100
    num_labeled = 100

    trainloader = MNISTTrainSet(num_labeled, batch_size)
    testloader = MNISTTestSet()
    testimages, testlabels = testloader.next_batch()

    ladder = Ladder(layer_sizes, denoising_cost).to(device)

    EPOCHES = 150
    LR = 0.02
    fit(ladder, trainloader, EPOCHES, LR)
