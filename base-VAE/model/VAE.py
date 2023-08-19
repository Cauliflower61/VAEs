import torch
import torch.nn as nn


class Vae(nn.Module):
    def __init__(self, in_channels, latent_dim, out_channels, img_size, hidden_channels=None):
        super().__init__()
        self.lantent_dim = latent_dim

        module = []
        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256, 512]

        # Encoder Building #
        for channels in hidden_channels:
            module.append(nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3,
                                                  stride=2, padding=1),
                                        nn.BatchNorm2d(channels),
                                        nn.LeakyReLU()))
            in_channels = channels

        self.pixel = int((img_size / (2 ** (len(hidden_channels)))) ** 2)

        self.encoder = nn.Sequential(*module)
        self.fc_mu = nn.Linear(self.pixel * hidden_channels[-1], latent_dim)
        self.fc_var = nn.Linear(self.pixel * hidden_channels[-1], latent_dim)

        # Decoder Building #
        module = []

        self.decoder_input = nn.Linear(latent_dim, self.pixel * hidden_channels[-1])
        hidden_channels.reverse()
        for i in range(len(hidden_channels) - 1):
            module.append(nn.Sequential(nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i+1],
                                                           kernel_size=3, stride=2, padding=1, output_padding=1),
                                        nn.BatchNorm2d(hidden_channels[i+1]),
                                        nn.LeakyReLU()))
        module.append(nn.Sequential(nn.ConvTranspose2d(hidden_channels[-1], hidden_channels[-1], kernel_size=3,
                                                       stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(hidden_channels[-1]),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=3, padding=1),
                                    nn.Tanh()))
        self.decoder = nn.Sequential(*module)

    def encode(self, x):
        out = self.encoder(x)
        vec = torch.flatten(self.encoder(x), start_dim=1)
        mu = self.fc_mu(vec)
        log_var = self.fc_var(vec)
        return mu, log_var

    def reparameter(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        sample = torch.randn_like(std)
        return std * sample + mu

    def decode(self, z):
        img = self.decoder_input(z)
        batch = z.shape[0]
        height = int(self.pixel ** 0.5)
        img = img.view(batch, -1, height, height)
        img = self.decoder(img)
        return img

    def forward(self, x):
        mu, log_var = Vae.encode(self, x)
        vec = Vae.reparameter(self, mu, log_var)
        img = Vae.decode(self, vec)
        return img, mu, log_var

    def loss_fn(self, truth, pre, mu, log_var):
        MSE_loss = nn.MSELoss()
        recon_loss = MSE_loss(truth, pre)
        kl_loss = torch.mean(-0.5 * torch.sum(1+log_var-mu**2-log_var.exp(), dim=1), dim=0)
        loss = recon_loss + 0.00025 * kl_loss
        return loss, recon_loss, kl_loss

    def sample(self, sample_size, temp):
        z = temp * torch.randn(size=(sample_size, self.lantent_dim)).to('cuda:0')
        generate = Vae.decode(self, z)
        return generate
