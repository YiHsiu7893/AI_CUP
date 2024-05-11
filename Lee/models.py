import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

# implement a diffusion model for conditional image generation.
# input will be a simple black and white image of how the road looks like
# output should be a RGB image of a satelite image of the road based on the input image
# image size is 428x240

# the model will be based on the diffusion model, which is a generative model that can generate images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Resnet_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Resnet_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, en_x):
        # Determine the number of input channels for conv1
        # This should be the sum of channels from x and en_x
        in_channels = x.size(1) + en_x.size(1)

        # Concatenate tensors along the channel dimension
        x = torch.cat([en_x, x], dim=1)  # Concatenating along the channel dimension (dim=1)

        # Adjust conv1 to accept the correct number of input channels
        self.conv1 = nn.Conv2d(in_channels, self.conv1.out_channels, kernel_size=3, stride=1, padding=1)

        # Forward pass through the layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class Unet(nn.Module):
    def __init__(self, out_channel):
        super(Unet, self).__init__()
        self.out_channel = out_channel
        # Encoder
        self.en_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.en_block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.en_block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.en_block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.middle = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        # Decoder
        self.de_upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.de_block1 = nn.Sequential(nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU())
        self.de_upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.de_block2 = nn.Sequential(nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU())
        self.de_upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.de_block3 = nn.Sequential(nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU())
        self.de_upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.de_block4 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        self.output_layer = nn.Conv2d(64, self.out_channel, kernel_size=1)

    def forward(self, x):
        x = x.float()
        print("x:", x.shape)
        en1 = self.en_block1(x)
        print("en1:", en1.shape)
        en2 = self.en_block2(en1)
        print("en2:", en2.shape)
        en3 = self.en_block3(en2)
        print("en3:", en3.shape)
        en4 = self.en_block4(en3)
        print("en4:", en4.shape)

        middle = self.middle(en4)
        middle = self.de_upconv1(middle)
        print("mid:", middle.shape)

        de1 = self.de_block1(torch.cat([middle, en4], dim=1))
        print("de1:", de1.shape)
        de1 = self.de_upconv2(de1)
        print("de1:", de1.shape)
        de2 = self.de_block2(torch.cat([de1, en3], dim=1))
        print("de2:", de2.shape)
        de2 = self.de_upconv3(de2)
        print("de2:", de2.shape)
        de3 = self.de_block3(torch.cat([de2, en2], dim=1))
        print("de3:", de3.shape)
        de3 = self.de_upconv4(de3)
        print("de3:", de3.shape)
        de4 = self.de_block4(torch.cat([de3, en1], dim=1))
        print("de4:", de4.shape)
        out = self.output_layer(de4)
        print("out:", out.shape)

        out = F.sigmoid(out)
        print(out.shape)

        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        return self.model(img)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False)]  # Modified kernel size and padding
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))  # Using BatchNorm2d instead of InstanceNorm2d
        layers.append(nn.ReLU(inplace=True))  # Changed activation function
        if dropout:
            layers.append(nn.Dropout2d(dropout))  # Changed dropout layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),  # Modified kernel size and stride
            nn.BatchNorm2d(out_size),  # Using BatchNorm2d instead of InstanceNorm2d
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))  # Changed dropout layer
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 32, normalize=False)  # Reduced from 64 to 32 channels
        self.down2 = UNetDown(32, 64)  # Reduced from 128 to 64 channels
        self.down3 = UNetDown(64, 128)  # Reduced from 256 to 128 channels
        self.down4 = UNetDown(128, 256, dropout=0.5)  # Reduced from 512 to 256 channels
        self.down5 = UNetDown(256, 256, dropout=0.5)  # Reduced from 512 to 256 channels
        self.down6 = UNetDown(256, 256, dropout=0.5)  # Reduced from 512 to 256 channels

        self.up1 = UNetUp(256, 256, dropout=0.5)  # Reduced from 512 to 256 channels
        self.up2 = UNetUp(512, 256, dropout=0.5)  # Reduced from 1024 to 256 channels
        self.up3 = UNetUp(512, 128, dropout=0.5)  # Reduced from 1024 to 128 channels
        self.up4 = UNetUp(256, 64, dropout=0.5)  # Reduced from 1024 to 64 channels

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        return self.final(u4)
