import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import glob
import random
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])

        if np.random.random() < 0.5:
            img = Image.fromarray(np.array(img)[:, ::-1, :], "RGB")

        img= self.transform(img)

        return img

    def __len__(self):
        return len(self.files)


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=1, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
generator = GeneratorUNet()
generator.load_state_dict(torch.load("saved_models/%s/gancut_model/generator_50.pth" % opt.dataset_name))
discriminator = Discriminator()
discriminator.load_state_dict(torch.load("saved_models/%s/gancut_model/discriminator_50.pth" % opt.dataset_name))

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, mode='test_selfdraw'),
    batch_size=opt.batch_size,
    shuffle=True,
    # num_workers=opt.n_cpu,
)

test_dataloader = DataLoader(
    ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, mode="test_selfdraw"),
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(i,imgs):
    """Saves a generated sample from the validation set"""
    fake_img = generator(imgs)
    img_sample = torch.cat((imgs.data, fake_img.data), -1)
    save_image(img_sample, "images/%s/test_image_selfdraw_cutout/test_%s.png" % (opt.dataset_name, i), nrow=1, normalize=True)

# ----------
#  Test self_draw
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(test_dataloader):
        # Determine approximate time left
        #save_image(batch, "images/%s/test_image_selfdraw/batch_%s.png" % (opt.dataset_name, i) ,normalize=True)
        batches_done = epoch * len(dataloader) + i

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done,batch)
