import torch
import torch.nn as nn
import torch.nn.functional as F

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
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

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
    
# class Resnet_Block(nn.Module): # this is a bottleneck block
#     def __init__(self, in_plane, planes, stride=1, downsample=None):
#         super(Resnet_Block, self).__init__()
#         self.conv1 = nn.Conv2d(in_plane, planes, kernel_size=1, stride=stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1)
#         self.bn3 = nn.BatchNorm2d(planes*4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)
        
#         out += identity
#         out = self.relu(out)

#         return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=2, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.cbam1 = CBAM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.cbam2 = CBAM(out_channels)

    def forward(self, x, en_x):
        # print("decoder")
        x = torch.cat([en_x, x], dim=1)
        # print(x.shape, en_x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu1(x)
        # print(x.shape)
        x = self.cbam1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu2(x)
        # print(x.shape)
        x = self.cbam2(x)
        # print(x.shape)
        return x
    
# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU(inplace=True)
    
#     def forward(self, x, en_x):
#         x = torch.cat([en_x, x], dim=1)
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         return x

class ResNetUNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetUNetGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder1 = nn.Sequential(
            Resnet_Block(self.in_channels, self.in_channels, 1),
            Resnet_Block(self.in_channels, self.in_channels, 1),
            Resnet_Block(self.in_channels, self.in_channels, 1)
        )
        self.encoder2 = nn.Sequential(
            Resnet_Block(self.in_channels, self.in_channels*2, 2),
            Resnet_Block(self.in_channels*2, self.in_channels*2, 1),
            Resnet_Block(self.in_channels*2, self.in_channels*2, 1),
            Resnet_Block(self.in_channels*2, self.in_channels*2, 1)
        )
        self.encoder3 = nn.Sequential(
            Resnet_Block(self.in_channels*2, self.in_channels*4, 2),
            Resnet_Block(self.in_channels*4, self.in_channels*4, 1),
            Resnet_Block(self.in_channels*4, self.in_channels*4, 1),
            Resnet_Block(self.in_channels*4, self.in_channels*4, 1),
            Resnet_Block(self.in_channels*4, self.in_channels*4, 1),
            Resnet_Block(self.in_channels*4, self.in_channels*4, 1)
        )
        self.encoder4 = nn.Sequential(
            Resnet_Block(self.in_channels*4, self.in_channels*8, 2),
            Resnet_Block(self.in_channels*8, self.in_channels*8, 1),
            Resnet_Block(self.in_channels*8, self.in_channels*8, 1)
        )

        self.de_upconv = nn.ConvTranspose2d(self.in_channels*8, self.in_channels*4, kernel_size=1, stride=1, padding=0)
        self.de_upconv1 = nn.ConvTranspose2d(self.in_channels*8+self.in_channels*4, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = DecoderBlock(128 + self.in_channels*4, self.in_channels*4)
        self.de_upconv2 = nn.ConvTranspose2d(self.in_channels*4, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = DecoderBlock(64 + self.in_channels*2, self.in_channels*2)
        self.de_upconv3 = nn.ConvTranspose2d(self.in_channels*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = DecoderBlock(32 + self.in_channels, self.in_channels)
        self.de_upconv4 = nn.ConvTranspose2d(self.in_channels, 32, kernel_size=2, stride=2)
        self.de_upconv5 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.out_layer = nn.Conv2d(32, self.out_channels, kernel_size=1)
        
    def forward(self, x):
        x = x[:, :3, :, :]
        # print("x",x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print("x",x.shape)

        en1 = self.encoder1(x)
        # print("en1",en1.shape)
        en2 = self.encoder2(en1)
        # print("en2",en2.shape)
        en3 = self.encoder3(en2)
        # print("en3",en3.shape)
        en4 = self.encoder4(en3)
        # print("en4",en4.shape)

        en4up = self.de_upconv(en4)
        # print("en4up",en4up.shape)
        mid = torch.cat([en4up, en4], dim=1)
        # print("mid",mid.shape)

        deu1 = self.de_upconv1(mid)
        # print("deu1",deu1.shape)
        de1 = self.decoder1(deu1, en3)
        # print("de1",de1.shape)
        deu2 = self.de_upconv2(de1)
        # print("deu2",deu2.shape)
        de2 = self.decoder2(deu2, en2)
        # print("de2",de2.shape)
        deu3 = self.de_upconv3(de2)
        # print("deu3",deu3.shape)
        de3 = self.decoder3(deu3, en1)
        # print("de3",de3.shape)
        deu4 = self.de_upconv4(de3)
        # print("deu4",deu4.shape)
        deu5 = self.de_upconv5(deu4)
        # print("deu5",deu5.shape)
        out = self.out_layer(deu5)
        # print("out",out.shape)

        # out = F.sigmoid(out)

        return out

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path}")

if __name__ == "__main__":
    model = ResNetUNetGenerator(1)
    model.to(device)
    # print parameter size use torch info
    print(sum(p.numel() for p in model.parameters()))