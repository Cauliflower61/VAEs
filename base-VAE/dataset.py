import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MyData(Dataset):
    def __init__(self,
                 train_npy,
                 valid_npy,
                 image_size,
                 flag="train"):
        super(MyData, self).__init__()
        # load data
        self.dataset = np.load(train_npy) if flag == "train" else np.load(valid_npy)
        self.image_size = image_size

    def __getitem__(self, item):
        image = self.dataset[item]
        image = cv2.resize(image, (self.image_size, self.image_size))
        return torch.from_numpy(image/image.max()).float().unsqueeze(dim=0)

    def __len__(self):
        return self.dataset.shape[0]


if __name__ == "__main__":
    t_npy = './datasets/MNIST_train.npy'
    v_npy = './datasets/MNIST_valid.npy'
    image_size = 64
    train_dataset = MyData(t_npy, v_npy, image_size)
    plt.imshow(train_dataset[0].squeeze(dim=0).numpy(), cmap='gray')
    plt.show()