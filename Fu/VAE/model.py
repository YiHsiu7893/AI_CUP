import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.input_dim = 1
        self.latent_dim = 100

        # Encoder layers
        self.encoder_1 = self.en_layer(1, 32)
        self.encoder_2 = self.en_layer(32, 64)
        self.encoder_3 = self.en_layer(64, 128)
        self.encoder_4 = self.en_layer(128, 256)

        # Latent
        self.fc_mu = nn.Linear(256*4*4, self.latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, self.latent_dim)

        # Decoder layers
        self.decoder_1 = self.de_layer(self.latent_dim, 256*4*4)
        self.decoder_2 = self.de_layer(256, 128)
        self.decoder_3 = self.de_layer(128, 64)

    def en_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def de_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 3, 2, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        # Encoder
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        x = self.encoder_4(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        x = self.decoder_1(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x, mu, logvar
    
    def generate(self, z):
        x = self.decoder_1(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        return self.generate(z)
    
    def loss(self, x, x_recon, mu, logvar):
        recon_loss = nn.MSELoss(reduction='sum')(x_recon, x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self
    
class img_dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        return x, 0
    

def train(model, train_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss = model.loss(x, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')
    return model

def test(model, test_loader):
    model.eval()
    for i, (x, _) in enumerate(test_loader):
        x_recon, mu, logvar = model(x)
        loss = model.loss(x, x_recon, mu, logvar)
        print(f'Iteration {i}, Loss: {loss.item()}')
    return model

def main():
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Assuming you have a train_loader and test_loader
    train_loader = None
    test_loader = None
    model = train(model, train_loader, optimizer, num_epochs=10)
    model = test(model, test_loader)
    model.save_model('vae.pth')
    model.load_model('vae.pth')
    model = model.eval()
    num_samples = 10
    samples = model.sample(num_samples)
    print(samples.shape)