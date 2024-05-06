import numpy as np 
import matplotlib.pyplot as plt 
import os, time, pickle, json 
from glob import glob 
from PIL import Image
import cv2 
from typing import List, Tuple, Dict
from statistics import mean 
from tqdm import tqdm 
import torch.nn.functional as F
import torch 
import torch.nn as nn 
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader 

MEAN = (0.5, 0.5, 0.5,)
STD = (0.5, 0.5, 0.5,)
RESIZE_ = (240, 428)
RESIZE = (128, 128)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def read_path(root_path):
    img_path = os.path.join(root_path, "img")
    label_path = os.path.join(root_path, "label_img")
    img_files = sorted(glob(os.path.join(img_path, "*.jpg")))
    label_files = sorted(glob(os.path.join(label_path, "*.png")))
    return list(zip(img_files, label_files))

def split_train_val(files: List[Tuple[str, str]], split_ratio: float = 0.8) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    return train_files, val_files


class Transform():
    def __init__(self, resize=RESIZE, mean=MEAN, std=STD):
        self.data_transform = transforms.Compose([
            transforms.Resize((RESIZE_[0] // 4, RESIZE_[1] // 4)),  # 将图像大小缩小为原始大小的四分之一
            transforms.Pad(padding=(0, 0, resize[1]-RESIZE_[1] // 4, resize[0]-RESIZE_[0] // 4)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
        
    def __call__(self, img: Image.Image):
        return self.data_transform(img)

    
class Dataset(object):
    def __init__(self, files: List[Tuple[str, str]]):
        self.files = files 
        self.transformer = Transform()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        output_path, input_path= self.files[idx]
        input_img = Image.open(input_path)
        output_img = Image.open(output_path)
        
        
        input_tensor = self.transformer(input_img)
        output_tensor = self.transformer(output_img)
        return input_tensor, output_tensor 

    def __len__(self):
        return len(self.files)
    
def show_img_sample(img: torch.Tensor, img1: torch.Tensor):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    ax = axes.ravel()
    ax[0].imshow(img.permute(1, 2, 0))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("input image", c="g")
    ax[1].imshow(img1.permute(1, 2, 0))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("label image", c="g")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.enc1 = self.conv2Relu(3, 32, 5)
        self.enc2 = self.conv2Relu(32, 64, pool_size=2)
        self.enc3 = self.conv2Relu(64, 128, pool_size=2)
        self.enc4 = self.conv2Relu(128, 256, pool_size=2)
        self.enc5 = self.conv2Relu(256, 512, pool_size=2)
        self.enc6 = self.conv2Relu(512, 1024, pool_size=2)
        
        
        self.dec000 = self.deconv2Relu(1024, 512, pool_size=2)
        self.dec00 = self.deconv2Relu(512+512, 256, pool_size=2)
        self.dec0 = self.deconv2Relu(256+256, 128, pool_size=2)
        self.dec1 = self.deconv2Relu(128+128, 64, pool_size=2)
        self.dec2 = self.deconv2Relu(64+64, 32, pool_size=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32+32, 3, 5, padding=2), 
            nn.Tanh()
        )
        
        
        # self.enc1 = self.conv2Relu(3, 32, 5)
        # self.enc2 = self.conv2Relu(32, 64, pool_size=2)
        # self.enc3 = self.conv2Relu(64, 128, pool_size=2)
        # self.enc4 = self.conv2Relu(128, 256, pool_size=1)
        
        # self.dec1 = self.deconv2Relu(256, 128, pool_size=1)
        # self.dec2 = self.deconv2Relu(128+128, 64, pool_size=2)
        # self.dec3 = self.deconv2Relu(64+64, 32, pool_size=2)
        # self.dec4 = nn.Sequential(
        #     nn.Conv2d(32+32, 3, 5, padding=2), 
        #     nn.Tanh()
        # )
        
    def conv2Relu(self, in_c, out_c, kernel_size=3, pool_size=None):
        layer = []
        if pool_size:
            # Down width and height
            layer.append(nn.AvgPool2d(pool_size))
        # Up channel size 
        layer.append(nn.Conv2d(in_c, out_c, kernel_size, padding=(kernel_size-1)//2))
        layer.append(nn.LeakyReLU(0.2, inplace=True))
        layer.append(nn.BatchNorm2d(out_c))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Dropout2d(p=0.2))
        return nn.Sequential(*layer)
    
    def deconv2Relu(self, in_c, out_c, kernel_size=3, stride=1, pool_size=None):
        layer = []
        if pool_size:
            # Up width and height
            layer.append(nn.UpsamplingNearest2d(scale_factor=pool_size))
        # Down channel size 
        layer.append(nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1))
        layer.append(nn.BatchNorm2d(out_c))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Dropout2d(p=0.2))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        # torch.Size([16, 3, 240, 428])
        # torch.Size([16, 32, 240, 428])
        # torch.Size([16, 64, 120, 214])
        # torch.Size([16, 128, 60, 107])
        # torch.Size([16, 256, 30, 53])
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        # print(x.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape, "\n\n")
        out = self.dec000(x6)
        # print(out.shape)
        # 计算需要填充的像素数目
        # padding_needed = x3.shape[-1] - out.shape[-1]

        # 在 out 的最后一维上进行填充
        # out = F.pad(out, (0, padding_needed))
        out = self.dec00(torch.cat((out, x5), dim=1)) # concat channel 
        out = self.dec0(torch.cat((out, x4), dim=1)) # concat channel 
        out = self.dec1(torch.cat((out, x3), dim=1)) # concat channel 
        out = self.dec2(torch.cat((out, x2), dim=1)) # concat channel 
        # print(out.shape)
        out = self.dec3(torch.cat((out, x1), dim=1))
        # print(out.shape)
        # print(out.shape)
        return out 
    # def forward(self, x):
    #     # torch.Size([16, 3, 240, 428])
    #     # torch.Size([16, 32, 240, 428])
    #     # torch.Size([16, 64, 120, 214])
    #     # torch.Size([16, 128, 60, 107])
    #     # torch.Size([16, 256, 30, 53])
    #     x1 = self.enc1(x)
    #     x2 = self.enc2(x1)
    #     x3 = self.enc3(x2)
    #     x4 = self.enc4(x3) # (b, 256, 4, 4)
    #     # print(x.shape)
    #     # print(x1.shape)
    #     # print(x2.shape)
    #     # print(x3.shape)
    #     # print(x4.shape, "\n\n")
    #     out = self.dec1(x4)
    #     # print(out.shape)
    #     # 计算需要填充的像素数目
    #     # padding_needed = x3.shape[-1] - out.shape[-1]

    #     # 在 out 的最后一维上进行填充
    #     # out = F.pad(out, (0, padding_needed))
    #     out = self.dec2(torch.cat((out, x3), dim=1)) # concat channel 
    #     # print(out.shape)
    #     out = self.dec3(torch.cat((out, x2), dim=1))
    #     # print(out.shape)
    #     out = self.dec4(torch.cat((out, x1), dim=1)) # (b, 3, 64, 64)
    #     # print(out.shape)
    #     return out 
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = self.conv2relu(6, 16, 5, cnt=1)
        self.layer2 = self.conv2relu(16, 32, pool_size=4)
        self.layer3 = self.conv2relu(32, 64, pool_size=2)
        self.layer4 = self.conv2relu(64, 128, pool_size=2)
        self.layer5 = self.conv2relu(128, 256, pool_size=2)
        self.layer6 = nn.Conv2d(256, 1, kernel_size=1)
        
    def conv2relu(self, in_c, out_c, kernel_size=3, pool_size=None, cnt=2):
        layer = []
        for i in range(cnt):
            if i == 0 and pool_size != None:
                # Down width and height 
                layer.append(nn.AvgPool2d(pool_size))
            # Down channel size 
            layer.append(nn.Conv2d(in_c if i == 0 else out_c, 
                                   out_c,
                                   kernel_size,
                                   padding=(kernel_size-1)//2))
            layer.append(nn.BatchNorm2d(out_c))
            layer.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layer)
        
    def forward(self, x, x1):
        x = torch.cat((x, x1), dim=1)
        out = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        return self.layer6(out) # (b, 1, 2, 2)
    
def train_fn(train_dl, G, D, criterion_bce, criterion_mae, optimizer_g, optimizer_d):
    G.to(device)
    D.to(device)
    
    G.train()
    D.train()
    LAMBDA = 100.0
    total_loss_g, total_loss_d = [], []
    for i, (input_img, real_img) in enumerate(tqdm(train_dl)):
        input_img = input_img.to(device)
        real_img = real_img.to(device)
        
        real_label = torch.ones(input_img.size()[0], 1, 4, 4, device=device)
        fake_label = torch.zeros(input_img.size()[0], 1, 4, 4, device=device)
        
        
        # Generator 
        fake_img = G(input_img)
        #print("fake_img: ", fake_img.shape)
        
        fake_img_ = fake_img.detach() # commonly using 
        #print("fake_img: ", fake_img.shape)
        out_fake = D(fake_img, input_img)
        #print("out_fake: ", out_fake.shape)
        loss_g_bce = criterion_bce(out_fake, real_label) # binaryCrossEntropy
        loss_g_mae = criterion_mae(fake_img, real_img) # MSELoss
        loss_g = loss_g_bce + LAMBDA * loss_g_mae 
        total_loss_g.append(loss_g.item())
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_g.backward(retain_graph=True)
        optimizer_g.step()
        # Discriminator
        out_real = D(real_img, input_img)
        loss_d_real = criterion_bce(out_real, real_label)
        out_fake = D(fake_img_, input_img)
        loss_d_fake = criterion_bce(out_fake, fake_label)
        loss_d = loss_d_real + loss_d_fake 
        total_loss_d.append(loss_d.item())
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
    return mean(total_loss_g), mean(total_loss_d), fake_img.detach().cpu() 


# def saving_img(fake_img, e):
#     os.makedirs("generated", exist_ok=True)
#     save_image(fake_img, f"generated/fake{str(e)}.png", range=(-1.0, 1.0), normalize=True)

def saving_img(fake_img, e):
    os.makedirs("generated", exist_ok=True)
    # 将像素值范围调整到0到1之间
    fake_img = (fake_img + 1) / 2
    save_image(fake_img, f"generated/fake{str(e)}.jpg", normalize=True)
    

def saving_logs(result):
    with open("train.pkl", "wb") as f:
        pickle.dump([result], f)
        
def saving_model(D, G, e):
    os.makedirs("weight", exist_ok=True)
    torch.save(G.state_dict(), f"weight/G{str(e+1)}.pth")
    torch.save(D.state_dict(), f"weight/D{str(e+1)}.pth")
        
def show_losses(g, d):
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    ax = axes.ravel()
    ax[0].plot(np.arange(len(g)).tolist(), g)
    ax[0].set_title("Generator Loss")
    ax[1].plot(np.arange(len(d)).tolist(), d)
    ax[1].set_title("Discriminator Loss")
    plt.show()


def train_loop(train_dl, G, D, num_epoch, lr=0.0002, betas=(0.5, 0.999)):

    G.to(device)
    D.to(device)
    optimizer_g = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_d = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    criterion_mae = nn.L1Loss()
    criterion_bce = nn.BCEWithLogitsLoss()
    total_loss_d, total_loss_g = [], []
    result = {}
    
    for e in range(num_epoch):
        loss_g, loss_d, fake_img = train_fn(train_dl, G, D, criterion_bce, criterion_mae, optimizer_g, optimizer_d)
        total_loss_d.append(loss_d)
        total_loss_g.append(loss_g)
        saving_img(fake_img, e+1)
        
        if e%10 == 0:
            saving_model(D, G, e)
        # 打印当前epoch数和对应的loss
        print(f"Epoch [{e+1}/{num_epoch}], Generator Loss: {loss_g:.4f}, Discriminator Loss: {loss_d:.4f}")
    try:
        result["G"] = total_loss_d 
        result["D"] = total_loss_g 
        saving_logs(result)
        show_losses(total_loss_g, total_loss_d)
        saving_model(D, G, e)
        print("successfully save model")
    finally:
        return G, D 

all_files = read_path("dataset/train")
train_files, val_files = split_train_val(all_files, split_ratio=0.8)
# print("Total set: ", len(all_files))
# print("Train set: ", len(train_files))
# print("Validation set: ", len(val_files))

train_ds = Dataset(train_files)
val_ds = Dataset(val_files)

# 修改成正确的方法调用方式
# show_img_sample(train_ds.__getitem__(10)[0], train_ds.__getitem__(10)[1])

BATCH_SIZE = 32

torch.manual_seed(0)
np.random.seed(0)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

G = Generator()
D = Discriminator()

EPOCH = 200
trained_G, trained_D = train_loop(train_dl, G, D, EPOCH)


