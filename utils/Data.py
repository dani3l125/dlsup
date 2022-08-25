import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

is_debug = True


class DIV2KDataset(Dataset):
    def __init__(self, dir='.', type='train', transform=None, target_transform=None, is_debug=False):
        self.is_debug = is_debug
        if self.is_debug:
            self.image = torch.randn(3, 224, 224)
            self.target = torch.randn(3, 448, 448)
        self.type = type
        self.target_dir = os.path.join(dir, f'DIV2K/DIV2K_{type}_HR')
        self.img_dir = os.path.join(dir, f'DIV2K/DIV2K_{type}_LR_bicubic/X2')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(
            [name for name in os.listdir(self.target_dir) if os.path.isfile(os.path.join(self.target_dir, name))])

    def __getitem__(self, idx):
        if self.is_debug:
            return self.image, self.target
        real_idx = idx + 801 if self.type == 'valid' else idx + 1
        img_path = os.path.join(self.img_dir, f'{str(real_idx).zfill(4)}x2.png')
        target_path = os.path.join(self.target_dir, f'{str(real_idx).zfill(4)}.png')
        image = read_image(img_path).type(torch.float32)
        target = read_image(target_path).type(torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        final_image = torch.zeros((3, 1020, 1020))
        final_target = torch.zeros((3, 2040, 2040))
        # Handling inconsistency of images size by zero padding
        final_image[:, :image.shape[1], :image.shape[2]] = image
        final_target[:, :target.shape[1], :target.shape[2]] = target
        return final_image, final_target


# visualize a sample of the data
def plot_data_grid(data, index_data, nRow, nCol, title=''):
    size_col = 2
    size_row = 2

    fig, axes = plt.subplots(nRow, nCol, constrained_layout=True, figsize=(nCol * size_col, nRow * size_row))

    data = data.detach().cpu().squeeze(axis=1)

    if nRow == 1:
        for j in range(nCol):
            index = index_data[j]

            im = np.zeros((data.shape[-2], data.shape[-1], 3))
            im[:, :, 0] = data[index, 0, :, :]
            im[:, :, 1] = data[index, 1, :, :]
            im[:, :, 2] = data[index, 2, :, :]
            axes.imshow(im, vmin=0, vmax=1)
            axes.xaxis.set_visible(False)
            axes.yaxis.set_visible(False)
    else:
        for i in range(nRow):
            for j in range(nCol):
                k = i * nCol + j
                index = index_data[k]
                im = np.zeros((data.shape[-2], data.shape[-1], 3))
                im[:, :, 0] = data[index, 0, :, :]
                im[:, :, 1] = data[index, 1, :, :]
                im[:, :, 2] = data[index, 2, :, :]
                axes[i, j].imshow(im, vmin=0, vmax=1)
                axes[i, j].xaxis.set_visible(False)
                axes[i, j].yaxis.set_visible(False)

    plt.title(title)
    plt.show()


def plot_data():
    pass
