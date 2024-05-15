import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.transforms import Lambda, Resize
from vae import Encoder, Decoder, VAE # Model

import datetime as dt
import pickle

class CustomDataset(Dataset):
    def __init__(self, data_folder, target_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.data_files = sorted(os.listdir(data_folder))
        self.target_folder = target_folder
        self.target_files = sorted(os.listdir(target_folder))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_name = self.data_files[idx]
        data_path = os.path.join(self.data_folder, data_name)
        target_name = self.target_files[idx]
        target_path = os.path.join(self.target_folder, target_name)

        # Load input data image
        data = Image.open(data_path).convert("RGB")

        # Assuming target images have the same filename but in a different folder
        target = Image.open(target_path).convert("RGB")

        if self.transform:
            data = self.transform(data)
            target = self.transform(target)

        return data, target


# Example usage
# Assuming you have your data and targets already loaded
data = "../train/label_img"
targets = "../train/img"

# Define the transformation
transform = transforms.Compose(
    [
        transforms.Resize((107, 60)),  # Resize the smallest edge to 128
        # transforms.Lambda(lambda img: pad_to_desired(img)),  # Pad the image if needed
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ]
)


def pad_to_desired(img):
    desired_size = 128
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
    return transforms.functional.pad(img, padding)


# Create an instance of your custom dataset
custom_dataset = CustomDataset(data_folder=data, target_folder=targets, transform=transform)

# Create a DataLoader
batch_size = 32
shuffle = True
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)


# Define loss function
MSE_loss_func = nn.MSELoss()

def loss_function(x, x_hat, mean, log_var):
    MSE = MSE_loss_func(x_hat, x.view(*x_hat.shape)) #x_hat, x.view(*x_hat.shape)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return MSE + KLD


# Initialize VAE model and move to GPU
input_dim = 107 * 60 * 3
hidden_dim = 107
latent_dim = 20
output_dim = 107 * 60 * 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
encoder = Encoder(input_dim, hidden_dim, latent_dim)
decoder = Decoder(latent_dim, hidden_dim, output_dim)
vae = VAE(encoder, decoder).to(device)
# vae = VAE(input_dim, hidden_dim, latent_dim).to(device)

# Define optimizer
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training loop
epochs = 1
train_losses = []

for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)  # Move data to GPU
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        # print(recon_batch.shape, data.shape, mu.shape, logvar.shape)
        # input("asdefew")
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print("Epoch: {} Average loss: {:.4f}".format(epoch + 1, train_loss / len(data_loader.dataset)))
    # Save the model
    train_losses.append(train_loss / len(data_loader.dataset))
    torch.save(vae.state_dict(), f"ckpt/vae_model_{epoch}.pth")

current_time = dt.datetime.now().strftime("%m%d-%H%M%S")
print(len(train_losses))
with open(f"save_data/losses_{current_time}_{epochs}.pkl", "wb") as f:
    pickle.dump(train_loss, f)


# Generate samples
vae.eval()
with torch.no_grad():
    num_samples = 10
    latent_samples = torch.randn(num_samples, latent_dim).to(device)  # Move latent samples to GPU
    generated_samples = vae.decode(latent_samples)
    print(generated_samples.shape, latent_samples.shape)
    generated_samples = generated_samples.view(-1, 3, 428, 240).cpu()  # Move generated samples back to CPU
# Convert the tensor to numpy
generated_samples_np = generated_samples.numpy()

# Plot the images
fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
for i, ax in enumerate(axs):
    # The transpose operation is needed because matplotlib expects images to be in the format (height, width, channels),
    # while PyTorch uses the format (channels, height, width)
    ax.imshow(generated_samples_np[i].transpose(1, 2, 0))
    ax.axis("off")  # Hide axes
plt.show()