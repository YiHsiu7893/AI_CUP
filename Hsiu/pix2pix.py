# Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 
import os, pickle
from glob import glob 
from PIL import Image
from statistics import mean 
from tqdm import tqdm 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader


# Preprocessing
MEAN = (0.5, 0.5, 0.5,)
STD = (0.5, 0.5, 0.5,)
PADSIZE = (428, 428)
RESIZE = (256, 256)


def read_path(root, type):
    root_path = root
    data = []

    if type == "Img":
        for img in glob(root_path + "img/" + "*.jpg"):
            data.append(img)

    elif type == "Edg":
        for img in glob(root_path + "label_img/" + "*.png"):
            data.append(img)
            
    return data


class Transform():
    def __init__(self, mean=MEAN, std=STD):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    def __call__(self, img: Image.Image):
        padded_img = zero_padding(self.data_transform(img), PADSIZE)
        resized_img = F.interpolate(padded_img.unsqueeze(0), size=RESIZE, mode='bilinear').squeeze(0)

        return resized_img


class Dataset(object):
    def __init__(self, files_x, files_y=None):
        self.files_x = files_x 
        self.files_y = files_y
        self.transform = Transform()
        
    def __getitem__(self, idx: int):
        img_x = Image.open(self.files_x[idx])
        input_tensor = self.transform(img_x)

        if self.files_y:
            img_y = Image.open(self.files_y[idx])
            target_tensor = self.transform(img_y)
        else:
            file_name = os.path.splitext(os.path.basename(self.files_x[idx]))[0]
            target_tensor = file_name

        return input_tensor, target_tensor
    
    def __len__(self):
        return len(self.files_x)

        
def show_img_sample(img: torch.Tensor, img1: torch.Tensor):
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    ax = axes.ravel()

    img = (img+1.0)/2.0
    img1 = (img1+1.0)/2.0

    ax[0].imshow(img.permute(1, 2, 0))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Edge")
    ax[1].imshow(img1.permute(1, 2, 0))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("Image")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    

def zero_padding(image, target_shape):
    height, width = target_shape
    _, original_height, original_width = image.shape
    
    # 計算 zero-padding 的上、下、左、右的大小
    pad_bottom = height - original_height
    pad_right = width - original_width
    
    # 在圖像周圍添加 zero-padding
    padded_image = np.pad(image, ((0, 0), (0, pad_bottom), (0, pad_right)), mode='constant')
    
    return torch.tensor(padded_image)

def crop_padding(img, target_size):
    # 提取目標大小
    target_height, target_width = target_size
    
    # 計算上下左右的填充大小
    original_height = img.size(1)
    original_width = img.size(2)
    
    # 裁剪圖像，保留原始部分
    cropped_img = img[:, :target_height, :original_width]
    return cropped_img


x = read_path("./Training dataset/", "Edg")
y = read_path("./Training dataset/", "Img")
test = read_path("./", "Edg")

train_dataset = Dataset(x, y)
test_dataset = Dataset(test)

"""
for i in range(10):
    edge, img = train_dataset.__getitem__(i)
    show_img_sample(edge, img)
"""

BATCH_SIZE = 16
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)
train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.enc1 = self.conv2Relu(3, 32, 5)
        self.enc2 = self.conv2Relu(32, 64, pool_size=4)
        self.enc3 = self.conv2Relu(64, 128, pool_size=2)
        self.enc4 = self.conv2Relu(128, 256, pool_size=2)
        
        self.dec1 = self.deconv2Relu(256, 128, pool_size=2)
        self.dec2 = self.deconv2Relu(128+128, 64, pool_size=2)
        self.dec3 = self.deconv2Relu(64+64, 32, pool_size=4)
        self.dec4 = nn.Sequential(
            nn.Conv2d(32+32, 3, 5, padding=2), 
            nn.Tanh()
        )

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
        return nn.Sequential(*layer)
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3) # (b, 256, 4, 4)
        
        out = self.dec1(x4)
        out = self.dec2(torch.cat((out, x3), dim=1)) # concat channel 
        out = self.dec3(torch.cat((out, x2), dim=1))
        out = self.dec4(torch.cat((out, x1), dim=1)) # (b, 3, 64, 64)
        return out
    

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
    G.train()
    D.train()
    LAMBDA = 100.0
    total_loss_g, total_loss_d = [], []
    for i, (input_img, real_img) in enumerate(tqdm(train_dl)):
        input_img = input_img.to(device)
        real_img = real_img.to(device)
        
        #real_label = torch.ones(input_img.size()[0], 1, 2, 2).to(device)
        #fake_label = torch.zeros(input_img.size()[0], 1, 2, 2).to(device)
        real_label = torch.ones(input_img.size()[0], 1, 8, 8).to(device)
        fake_label = torch.zeros(input_img.size()[0], 1, 8, 8).to(device)

        # Generator 
        fake_img = G(input_img)
        fake_img_ = fake_img.detach() # commonly using 
        out_fake = D(fake_img, input_img)

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

def saving_img(fake_img, e):
    os.makedirs("generated", exist_ok=True)
    save_image(fake_img, f"generated/fake{str(e)}.png", value_range=(-1.0, 1.0), normalize=True)
    
def saving_logs(result):
    with open("train.pkl", "wb") as f:
        pickle.dump([result], f)
        
def saving_model(D, G, e):
    os.makedirs("weight", exist_ok=True)

    torch.save(G.state_dict(), f"weight/G.pth")
    torch.save(D.state_dict(), f"weight/D.pth")
    with open("weight/iteration.txt", "w") as f:
        f.write(str(e))

def show_losses(g, d):
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    ax = axes.ravel()
    ax[0].plot(np.arange(len(g)).tolist(), g)
    ax[0].set_title("Generator Loss")
    ax[1].plot(np.arange(len(d)).tolist(), d)
    ax[1].set_title("Discriminator Loss")
    plt.show()

def train_loop(train_dl, G, D, num_epoch, iteration_count, lr=0.0002, betas=(0.5, 0.999)):
    G.train()
    D.train()
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
        
        if (iteration_count+e+1)%10 == 0:
            saving_img(fake_img, iteration_count+e+1)
    try:
        result["G"] = total_loss_d 
        result["D"] = total_loss_g
        saving_logs(result)
        show_losses(total_loss_g, total_loss_d)
        saving_model(D, G, iteration_count+e+1)
        print("successfully save model")

    finally:
        return G, D 


def load_G():
    G = Generator()
    G.load_state_dict(torch.load("weight/G.pth", map_location={"cuda:0": "cpu"}))

    with open("weight/iteration.txt", "r") as f:
        file_content = f.read()
        print(f"this is a pretrained model trained for {file_content} epochs...\n")
    return G, int(file_content)

def load_D():
    D = Discriminator()
    D.load_state_dict(torch.load(f"weight/D.pth", map_location={"cuda:0": "cpu"}))
    return D


#G = Generator()
#D = Discriminator()
#i = 0
G, i = load_G()
D = load_D()
EPOCH = 10
trained_G, trained_D = train_loop(train_dl, G, D, EPOCH, i)


def de_norm(img):
    img_ = img.mul(torch.FloatTensor(STD).view(3, 1, 1))
    img_ = img_.add(torch.FloatTensor(MEAN).view(3, 1, 1)).detach().numpy()
    img_ = np.transpose(img_, (1, 2, 0))

    return img_ 

def evaluate():
    with torch.no_grad():
        G, _ = load_G()
        G.eval()
        G.to(device)

        for i, (edge, file_name) in enumerate(tqdm(test_dl)):
            edge = edge.to(device)

            fake_img = G(edge)
            for i in range(fake_img.shape[0]):
                img = fake_img[i]
                resized_img = F.interpolate(img.unsqueeze(0), size=(428, 428), mode='bilinear', align_corners=False).squeeze(0)
                cropped_img = crop_padding(resized_img, (240, 428))

                plt.imsave(f'./result/{file_name[i]}.jpg', de_norm(cropped_img))
                
evaluate()
