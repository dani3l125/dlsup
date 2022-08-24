import torch
from utils.Data import DIV2KDataset, plot_data_grid
from torchvision import transforms
from Models import Unet
from torch.utils.data import DataLoader
import numpy as np
import argparse
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='configuration file path')
parser.add_argument('--cfg', type=str, default= 'cfg_dlc.yaml',
                    help='path to configuration file')

args = parser.parse_args()
with open(args.cfg, 'r') as stream:
    cfg = yaml.safe_load(stream)

DIV2K_PATH = cfg['DIV2K_PATH']

MODEL_PATH = cfg['MODEL_PATH']
NAME = cfg['NAME']

def train(model, visualize_data=False):
    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Resize(1020)
    ])
    target_transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Resize(2040)
    ])
    train_ds = DIV2KDataset(dir=DIV2K_PATH, transform=transform, target_transform=target_transform)
    val_ds = DIV2KDataset(dir=DIV2K_PATH, type='valid', transform=transform, target_transform=target_transform)

    train_dl = DataLoader(train_ds, batch_size=1, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=True)

    if visualize_data:
        nRow = 1
        nCol = 10
        index_data = np.arange(nRow * nCol)  # show only first images

        for index_batch, (im, target) in enumerate(train_dl):
            plot_data_grid(im, index_data, nRow, nCol, 'sample from augmented batch')
            plot_data_grid(target, index_data, nRow, nCol)

        nRow = 2
        nCol = 12
        index_data = np.arange(0, nRow * nCol)  # show only first images
        originals_train_list = []
        blurry_train_list = []
        for idx in index_data:
            originals_train, blurry_train = train_ds[idx]
            originals_train = originals_train[0]
            blurry_train = blurry_train[0]
            originals_train_list.append(originals_train)
            blurry_train_list.append(blurry_train)

        originals_train = torch.tensor(np.stack(originals_train_list, 0))
        blurry_train = torch.tensor(np.stack(blurry_train_list, 0))

        prediction_train = model(blurry_train.unsqueeze(dim=1).to(device))

        plot_data_grid(originals_train, index_data, nRow, nCol, title='original size images, train')
        plot_data_grid(blurry_train, index_data, nRow, nCol, title='input images, train')
        plot_data_grid(prediction_train, index_data, nRow, nCol, title='output images, train')

        nRow = 2
        nCol = 10
        originals_test_list = []
        blurry_test_list = []
        for idx in index_data:
            originals_test, blurry_test = val_ds[idx]
            originals_test = originals_test[0]
            blurry_test = blurry_test[0]
            originals_test_list.append(originals_test)
            blurry_test_list.append(blurry_test)

        originals_test = torch.tensor(np.stack(originals_test_list, 0))
        blurry_test = torch.tensor(np.stack(blurry_test_list, 0))

        prediction_test = model(blurry_test.unsqueeze(dim=1).to(device))

        plot_data_grid(originals_test, index_data, nRow, nCol, title='original size images, test')
        plot_data_grid(blurry_test, index_data, nRow, nCol, title='input images, test')
        plot_data_grid(prediction_test, index_data, nRow, nCol, title='output images, test')


if __name__ == '__main__':
    model = Unet().to(device).load_state_dict(f'./model_{NAME}.pth')
    train(model)
