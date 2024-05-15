import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.FC_input(x)
        print(x.shape)
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        x_hat = x_hat.view(x_hat.size(0), 3, 428, 240)
        return x_hat


class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        print(x_hat.shape)

        return x_hat, mean, log_var


# class VAE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(VAE, self).__init__()

#         # Encoder layers
#         self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
#         self.encoder_fc2_mean = nn.Linear(hidden_dim, latent_dim)
#         self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)

#         # Decoder layers
#         self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
#         self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

#     def encode(self, x):
#         x = F.relu(self.encoder_fc1(x))
#         mean = self.encoder_fc2_mean(x)
#         logvar = self.encoder_fc2_logvar(x)
#         return mean, logvar

#     def reparameterize(self, mean, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mean + eps * std

#     def decode(self, z):
#         z = F.relu(self.decoder_fc1(z))
#         return torch.sigmoid(self.decoder_fc2(z))

#     def forward(self, x):
#         mean, logvar = self.encode(x.view(-1, 107 * 60 * 3))
#         z = self.reparameterize(mean, logvar)
#         return self.decode(z), mean, logvar

# import os
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import matplotlib.pyplot as plt


# class CustomDataset(Dataset):
#     def __init__(self, data_folder, target_folder, transform=None):
#         self.data_folder = data_folder
#         self.transform = transform
#         self.data_files = sorted(os.listdir(data_folder))
#         self.target_folder = target_folder
#         self.target_files = sorted(os.listdir(target_folder))

#     def __len__(self):
#         return len(self.data_files)

#     def __getitem__(self, idx):
#         data_name = self.data_files[idx]
#         data_path = os.path.join(self.data_folder, data_name)
#         target_name = self.target_files[idx]
#         target_path = os.path.join(self.target_folder, target_name)

#         # Load input data image
#         data = Image.open(data_path).convert("RGB")

#         # Assuming target images have the same filename but in a different folder
#         target = Image.open(target_path).convert("RGB")

#         if self.transform:
#             data = self.transform(data)
#             target = self.transform(target)

#         return data, target


# # Example usage
# # Assuming you have your data and targets already loaded
# data = "train/label_img"
# targets = "train/img"

# # Define the transformation
# transform = transforms.Compose(
#     [
#         transforms.Resize(128),  # Resize the smallest edge to 128
#         transforms.Lambda(lambda img: pad_to_desired(img)),  # Pad the image if needed
#         transforms.ToTensor(),  # Convert the image to a PyTorch tensor
#     ]
# )


# def pad_to_desired(img):
#     desired_size = 128
#     delta_width = desired_size - img.size[0]
#     delta_height = desired_size - img.size[1]
#     padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
#     return transforms.functional.pad(img, padding)


# # Create an instance of your custom dataset
# custom_dataset = CustomDataset(data_folder=data, target_folder=targets, transform=transform)

# # Create a DataLoader
# batch_size = 32
# shuffle = True
# data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)


# # Define the VAE model
# class VAE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(VAE, self).__init__()

#         # Encoder layers
#         self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
#         self.encoder_fc2_mean = nn.Linear(hidden_dim, latent_dim)
#         self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)

#         # Decoder layers
#         self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
#         self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

#     def encode(self, x):
#         x = F.relu(self.encoder_fc1(x))
#         mean = self.encoder_fc2_mean(x)
#         logvar = self.encoder_fc2_logvar(x)
#         return mean, logvar

#     def reparameterize(self, mean, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mean + eps * std

#     def decode(self, z):
#         z = F.relu(self.decoder_fc1(z))
#         return torch.sigmoid(self.decoder_fc2(z))

#     def forward(self, x):
#         mean, logvar = self.encode(x.view(-1, 128 * 128 * 3))
#         z = self.reparameterize(mean, logvar)
#         return self.decode(z), mean, logvar


# # Define loss function
# def loss_function(recon_x, x, mu, logvar):
#     MSE = F.mse_loss(recon_x, x.view(-1, 128 * 128 * 3), reduction="sum")
#     # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 128 * 128 * 3), reduction="sum")
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return MSE + KLD


# # Initialize VAE model and move to GPU
# input_dim = 128 * 128 * 3
# hidden_dim = 256
# latent_dim = 20
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vae = VAE(input_dim, hidden_dim, latent_dim).to(device)

# # Define optimizer
# optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# # Training loop
# epochs = 1
# for epoch in range(epochs):
#     vae.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(data_loader):
#         data = data.to(device)  # Move data to GPU
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = vae(data)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#     print("Epoch: {} Average loss: {:.4f}".format(epoch + 1, train_loss / len(data_loader.dataset)))
#     # Save the model
#     # torch.save(vae.state_dict(), f"VAE/ckpt/vae_model_{epoch}.pth")


# # Generate samples
# vae.eval()
# with torch.no_grad():
#     num_samples = 10
#     latent_samples = torch.randn(num_samples, latent_dim).to(device)  # Move latent samples to GPU
#     generated_samples = vae.decode(latent_samples).view(-1, 3, 128, 128).cpu()  # Move generated samples back to CPU
# # Convert the tensor to numpy
# generated_samples_np = generated_samples.numpy()

# # Plot the images
# fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
# for i, ax in enumerate(axs):
#     # The transpose operation is needed because matplotlib expects images to be in the format (height, width, channels),
#     # while PyTorch uses the format (channels, height, width)
#     ax.imshow(generated_samples_np[i].transpose(1, 2, 0))
#     ax.axis("off")  # Hide axes
# plt.show()
