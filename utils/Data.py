import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class DIV2KDataset(Dataset):
    def __init__(self, dir='.', type='train' , transform=None, target_transform=None):
        self.target_dir = os.path.join(dir, f'DIV2K/DIV2K_{type}_HR')
        self.img_dir = os.path.join(dir, f'DIV2K/DIV2K_{type}_LR_bicubic/X2')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len([name for name in os.listdir(self.target_dir) if os.path.isfile(os.path.join(self.target_dir, name))])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{str(idx).zfill(4)}x2.png')
        target_path = os.path.join(self.target_dir, f'{str(idx).zfill(4)}.png')
        image = read_image(img_path)
        target = read_image(target_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


# visualize a sample of the data
def plot_data_grid(data, index_data, nRow, nCol, title=''):
    size_col = 2
    size_row = 2

    fig, axes = plt.subplots(nRow, nCol, constrained_layout=True, figsize=(nCol * size_col, nRow * size_row))

    data = data.detach().cpu().squeeze(axis=1)

    if nRow == 1:
        for j in range(nCol):
            index = index_data[j]

            axes[j].imshow(data[index], cmap='gray', vmin=0, vmax=1)
            axes[j].xaxis.set_visible(False)
            axes[j].yaxis.set_visible(False)
    else:
        for i in range(nRow):
            for j in range(nCol):
                k = i * nCol + j
                index = index_data[k]

                axes[i, j].imshow(data[index], cmap='gray', vmin=0, vmax=1)
                axes[i, j].xaxis.set_visible(False)
                axes[i, j].yaxis.set_visible(False)

    plt.title(title)
    plt.show()