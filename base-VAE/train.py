import torch
from torch.utils.data import DataLoader
from dataset import MyData
from model.VAE import Vae
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import utils

# parameters definition #
t_npy = './datasets/MNIST_train.npy'
v_npy = './datasets/MNIST_valid.npy'
Checkpoint_dir = './Checkpoint/'
log_dir = './TensorBoardSave/'
device = 'cuda:0'
batch_size = 32
image_size = 64
latent_dim = 100
learning_rate = 5e-4
epoch_num = 10

# load data #
train_dataset = MyData(t_npy, v_npy, image_size)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
# TensorBoard #
writer = SummaryWriter(log_dir)
batch_num = len(train_dataloader)

# generate model #
net = Vae(1, latent_dim, 1, image_size).to(device)
optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

# model train #
net.train()
for epoch in range(epoch_num):
    total_loss = 0
    loop = tqdm(train_dataloader, desc='Train')
    for iter1, train_data in enumerate(loop):
        batch = train_data.shape[0]
        truth = train_data.to(device)
        optim.zero_grad()
        pre, mu, log_var = net(truth)
        loss, recon_loss, kl_loss = net.loss_fn(truth=truth, pre=pre, mu=mu, log_var=log_var)
        total_loss += loss.data * batch
        loss.backward()
        optim.step()
        global_step = epoch * batch_num + iter1
        writer.add_scalar(tag="training loss", scalar_value=loss,
                          global_step=global_step)
        loop.set_description(f'Epoch [{epoch + 1}/{epoch_num}]')
        loop.set_postfix(loss=loss.data.item(), recon_loss=recon_loss.data.item(), kl_loss=kl_loss.data.item())

        if global_step % 1000 == 0:
            with torch.no_grad():
                utils.save_image(
                    net.sample(sample_size=16, temp=1).detach().cpu().data,
                    f"sample/{str(global_step + 1).zfill(6)}.png",
                    normalize=True,
                    nrow=4,
                    range=(0, 1),
                )
    if (epoch+1) % 10 == 0:
        torch.save(net.state_dict(), Checkpoint_dir + 'epoch' + str(epoch+1) + '.pkl')
writer.close()