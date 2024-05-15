import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar
from tqdm import tqdm

# import wandb
from utils import *
from data_preprocess import *
from modules import UNet_conditional, EMA

class SSIM_loss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim_weight = weight
        self.SSIM = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, x, y):
        # if any of the pixels are nan, print a warning
        if torch.isnan(x).any():
            print("Warning: NaN in input x")
        if torch.isnan(y).any():
            print("Warning: NaN in input y")
        mse = self.mse(x, y)
        de_x = torch.clamp((x + 1) / 2, 0, 1)
        de_y = torch.clamp((y + 1) / 2, 0, 1)
        ssim = 1 - self.SSIM(de_x, de_y)
        return mse + self.ssim_weight * max(0,ssim)

class Diffusion(nn.Module):
    def __init__(self, noise_steps=1000, beta_start=0.0001, beta_end=0.02, device="cuda:0", steps=1000):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.model = UNet_conditional().to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False).to(device)
        self.ema = EMA(0.999)
        self.device = device

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = self.alpha.cumprod(dim=0)

        self.img_h = 96
        self.img_w = 96

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-4, steps_per_epoch=steps, epochs=args['epochs'], anneal_strategy="linear")
        self.criterion = SSIM_loss()
        self.scaler = torch.cuda.amp.GradScaler()

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def add_noise(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        # self.optimizer.step()

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0
        tqdm_data_loader = tqdm(data_loader)
        for i, (data, target) in enumerate(tqdm_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            # plot data and target
            # from matplotlib import pyplot as plt
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(data[0].cpu().permute(1, 2, 0))
            # ax[0].set_title("Data")
            # ax[1].imshow(target[0].cpu().permute(1, 2, 0))
            # ax[1].set_title("Target")
            # plt.show()
            t = torch.randint(1, self.noise_steps, (data.size(0),), device=self.device)
            with torch.cuda.amp.autocast():
                noisy_data, noise = self.add_noise(data, t)
                pred_noise = self.model(noisy_data, t)
                if torch.isnan(pred_noise).any():
                    print("Warning: NaN in pred_noise")
                    print(f"t: {t}")
                    print(f"noisy_data: {noisy_data}")
                    print(f"pred_noise: {pred_noise}")
                    input("!!")
                loss = self.criterion(noise, pred_noise)
                total_loss += loss.item()
            self.train_step(loss)

            tqdm_data_loader.set_postfix(loss=loss.item())

        return total_loss / len(data_loader.dataset)
    
    def val_epoch(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            tqdm_data_loader = tqdm(data_loader)
            for i, (data, target) in enumerate(tqdm_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                t = torch.randint(0, self.noise_steps, (data.size(0),), device=self.device)
                noisy_data, noise = self.add_noise(data, t)
                pred_noise = self.model(noisy_data, t)
                loss = self.criterion(noise, pred_noise)
                total_loss += loss.item()

        return total_loss / len(data_loader.dataset)
    
    def fit(self, train_loader, val_loader, epochs):
        best_loss = float('inf')
        losses = []
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            losses.append(train_loss)
            # val_loss = self.val_epoch(val_loader)
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f}") # Val Loss: {val_loss:.4f}
            if train_loss < best_loss:
                best_loss = train_loss
                import datetime as dt
                current_time = dt.datetime.now().strftime("%m%d-%H%M%S")
                self.save(f"save_data/{current_time}_ep{epoch}_{train_loss}_dm.pth")

        plot_loss(losses, epochs)

    # inference: input a noisy image and return a denoised image
    def inference(self, noisy_img, t):
        self.model.eval()
        with torch.no_grad():
            noisy_img = noisy_img.to(self.device)
            t = torch.tensor([t], device=self.device)
            return self.model(noisy_img, t)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    # parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    # parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    # parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default="../dataset", help='path to dataset')
    parser.add_argument('--train_folder', type=str, default="train", help='train folder')
    parser.add_argument('--device', type=str, default="cuda:0", help='device')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    # parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=1000, help='noise steps')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    args = vars(parser.parse_args())

    return args
        
def main(args):
    train_dataloader = get_data(args)
    print(f"Train Dataloader: {len(train_dataloader)}")
    diffuser = Diffusion(noise_steps=args["noise_steps"], device=args["device"], steps=len(train_dataloader))
    diffuser.fit(train_dataloader, train_dataloader, args["epochs"])

    test_path = "../dataset/test/label_img"
    test_dataset = LabelDataset(test_path, label_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test Dataloader: {len(test_loader)}")

    for i, (data, target) in enumerate(train_dataloader):
        if i == 0:
            # noisy_data, noise = diffuser.add_noise(data, 0)
            denoised_img = diffuser.inference(data, 1000)
            denoised_img = denoised_img.cpu()
            # plot the first image and its target
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(data[0].permute(1, 2, 0))
            ax[0].set_title("Noisy Data")
            ax[1].imshow(denoised_img[0].permute(1, 2, 0))
            ax[1].set_title("Denoised Data")
            ax[2].imshow(target[0].permute(1, 2, 0))
            ax[2].set_title("Target")
            plt.show()
            break

if __name__ == "__main__":
    args = parse_args()
    main(args)