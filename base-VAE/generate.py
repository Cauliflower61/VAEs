from model.VAE import Vae
import torch
from torchvision import utils


latent_dim = 100
img_size = 64
temp = 0.7

load_path = './Checkpoint/epoch10.pkl'
net = Vae(1, latent_dim, 1, 64).to('cuda:0')
net.load_state_dict(torch.load(load_path))

with torch.no_grad():
    generate = net.sample(15*15, temp).detach().cpu().data
    utils.save_image(generate, f'./generate.png', nrow=15, normalize=True, range=(0, 1))
