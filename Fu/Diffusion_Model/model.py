import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_preprocess import CustomDataset, get_data_loader
from datetime import datetime
import pickle

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

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x, en_x):
        x = torch.cat([en_x, x], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x
    

class Unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()
        self.out_c = out_channel
        self.in_c = in_channel
        self.in_conv = nn.DoubleConv(in_channel, 64)
        self.down1 = nn.MaxPool2d(2)



    # def forward(self, x):



    #     return out
    
class UNet_conditional(Unet):
    def __init__(self, )
    
def FID_loss(real_features, fake_features):
    import numpy as np
    mu_real = np.mean(real_features, dim=0)
    mu_fake = np.mean(fake_features, dim=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    mu_diff = torch.norm(mu_real - mu_fake)
    sigma_diff = torch.norm(sigma_real + sigma_fake - 2 * torch.sqrt(sigma_real * sigma_fake))
    return mu_diff + sigma_diff
    
def main():
    model = Unet(3).to(device)
    dataloader = get_data_loader()
    criterion = nn.MSELoss()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    num_epochs = 30

    losses = []

    from tqdm import tqdm
    for epoch in range(num_epochs):
        tqdm_loader = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]", dynamic_ncols=True)
        loss_sum = 0.0
        for i, (data, target) in enumerate(tqdm_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item()
            # fid_loss = FID_loss(data, output)
            loss.backward()
            optimizer.step()
            tqdm_loader.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']}) #, "FID_loss": fid_loss.item()
        lr_scheduler.step()

        losses.append(loss_sum/len(dataloader.dataset))

        if (epoch) % 10 == 0:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            torch.save(model.state_dict(), f"{current_time}_{epoch}_{loss_sum/len(dataloader.dataset)}_model.pth")

    with open(f"{num_epochs}_{lr}_losses.pkl", "wb") as f:
        pickle.dump(losses, f)

    sample_data, sample_target = next(iter(dataloader))
    sample_data, sample_target = sample_data.to(device), sample_target.to(device)
    sample_output = model(sample_data)
    print(sample_output.shape)
    print(sample_target.shape)
    # resize the output to 428x240
    sample_output = F.interpolate(sample_output, size=(240, 428), mode="bicubic")
    sample_target = F.interpolate(sample_target, size=(240, 428), mode="bicubic")
    sample_data = F.interpolate(sample_data, size=(240, 428), mode="bicubic")
    # plot both the target and the output
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(sample_output[0].cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].set_title("Output")
    ax[1].imshow(sample_target[0].cpu().detach().numpy().transpose(1, 2, 0))
    ax[1].set_title("Target image")
    ax[2].imshow(sample_data[0].cpu().detach().numpy().transpose(1, 2, 0))
    ax[2].set_title("Input image")
    plt.show()


if __name__ == "__main__":
    main()