import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from PIL import Image
from models import Discriminator, GeneratorUNet
from torch.utils.data import DataLoader
from dataset import ImageDataset
from torchvision.utils import save_image
import time
from tqdm import tqdm

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("ckpt", exist_ok=True)

criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

lambda_pixel = 100

patch = (1, 428 // 2**4, 240 // 2**4)

generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Configure dataloaders
transforms_ = transforms.Compose(
    [
        transforms.Resize((214, 120), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
dataset = ImageDataset("../train/train_real", "../train/train_target", transforms_=transforms_)
val_dataset = ImageDataset("../train/valid_real", "../train/valid_target", transforms_=transforms_, mode="val")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

prev_time = time.time()

num_epoch = 1
for epoch in range(num_epoch):
    tqdm_loader = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epoch}]", dynamic_ncols=True)
    for i, data in enumerate(tqdm_loader):
        # Training teh discirminator
        optimizer_D.zero_grad()

        real_images = data[0].to(device)
        fake_images = generator(real_images).detach()

        real_labels = torch.ones(real_images.size(0), 1, 30, 30).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1, 30, 30).to(device)

        real_outputs = discriminator(real_images)
        fake_outputs = discriminator(fake_images)

        real_loss = criterion_GAN(real_outputs, real_labels)
        fake_loss = criterion_GAN(fake_outputs, fake_labels)

        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        optimizer_D.STEP()

        # Training the generator
        optimizer_G.zero_grad()

        fake_images = generator(real_images)
        outputs = discriminator(fake_images)

        generator_loss = criterion_GAN(outputs, real_labels)
        generator_loss.backward()
        optimizer_G.step()
