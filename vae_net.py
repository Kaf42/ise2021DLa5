import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


dataset_name="facades"

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["B"].type(Tensor))
    real_B = Variable(imgs["A"].type(Tensor))
    fake_B, mu, logvar = net(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/vae_image/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)


def deprocess_img(x):
    return (x + 1.0) / 2.0


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28*28*3, 400)
        self.fc21 = nn.Linear(400, 20)  # 均值
        self.fc22 = nn.Linear(400, 20)  # 方差
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 28*28*3)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        x = F.tanh(self.fc4(h3))
        return x

    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        # if torch.cuda.is_available():
        # eps = torch.cuda.FloatTensor(std.size()).normal_()    # 从标准的正态分布中随机采样一个eps
        # else:
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        output = self.decoder(z)
        output = output.view(output.shape[0], 3, 28, 28)
        return output, mu, logvar

net = VAE()

reconstruction_function = nn.MSELoss(reduction='sum')


def loss_function(recon_x, x, mu, logvar):
    MSE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return MSE + KLD


# 使用 adam 来进行训练，学习率是 3e-4, beta1 是 0.5, beta2 是 0.999
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer

cuda = True if torch.cuda.is_available() else False
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
start_epoch=0

if start_epoch != 0:
    # Load pretrained models
    net.load_state_dict(torch.load("saved_models/%s/vae_model/vae_net_%d.pth" % (dataset_name, start_epoch)))
else:
    # Initialize weights
    net.apply(weights_init_normal)


def train_vae(net, optim, show_every=1000, num_epochs=100):
    iter_count = 0
    for epoch in range(start_epoch,num_epochs):
        for i, batch in enumerate(train_data):
            real_A = Variable(batch["B"].type(Tensor))
            img = Variable(batch["A"].type(Tensor))
           # print(img.shape)
            output, mu, logvar = net(img)
            loss = loss_function(output, img, mu, logvar) / img.size(0)

            optim.zero_grad()
            loss.backward()
            optim.step()  # 优化判别网络

            batches_done = epoch * len(train_data) + i

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(train_data) - 1,
                    loss.item(),
                )
            )

            if batches_done % 500 == 0:
                sample_images(batches_done)

            if  epoch % 50 == 0:
                # Save model checkpoints
                torch.save(net.state_dict(), "saved_models/%s/vae_model/vae_net_%d.pth" % (dataset_name, epoch))


transforms_ = [
    transforms.Resize((28, 28), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_data = DataLoader(
    ImageDataset("data/%s" % dataset_name, transforms_=transforms_),
    batch_size=10,
    shuffle=True,
    #num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("data/%s" % dataset_name, transforms_=transforms_, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=0,
)

optim = get_optimizer(net)

train_vae(net, optim, num_epochs=1001)
