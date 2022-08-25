import matplotlib.pyplot as plt
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
parser.add_argument('--cfg', type=str, default='cfg_ferrum.yaml',
                    help='path to configuration file')

args = parser.parse_args()
with open(args.cfg, 'r') as stream:
    cfg = yaml.safe_load(stream)

DIV2K_PATH = cfg['DIV2K_PATH']

MODEL_PATH = cfg['MODEL_PATH']
NAME = cfg['NAME']


# def inference(model, visualize_data=False):
#     transform = transforms.Compose([
#         #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         # transforms.Resize(1020)
#     ])
#     target_transform = transforms.Compose([
#         # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         # transforms.Resize(2040)
#     ])
#     train_ds = DIV2KDataset(dir=DIV2K_PATH, transform=transform, target_transform=target_transform)
#     val_ds = DIV2KDataset(dir=DIV2K_PATH, type='valid', transform=transform, target_transform=target_transform)
#
#     train_dl = DataLoader(train_ds, batch_size=1, num_workers=4, pin_memory=True)
#     val_dl = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=True)
#
#     if visualize_data:
#         nRow = 1
#         nCol = 1
#         index_data = np.arange(nRow * nCol)  # show only first images
#
#         # for index_batch, (im, target) in enumerate(train_dl):
#         #     plot_data_grid(im, index_data, nRow, nCol, 'sample from augmented batch')
#         #     plot_data_grid(target, index_data, nRow, nCol)
#         #
#         # nRow = 1
#         # nCol = 3
#         # index_data = np.arange(0, nRow * nCol)  # show only first images
#         originals_train_list = []
#         blurry_train_list = []
#         for idx in index_data:
#             originals_train, blurry_train = train_ds[idx]
#             originals_train = originals_train
#             blurry_train = blurry_train
#             originals_train_list.append(originals_train)
#             blurry_train_list.append(blurry_train)
#
#         originals_train = torch.tensor(np.stack(originals_train_list, 0))
#         blurry_train = torch.tensor(np.stack(blurry_train_list, 0))
#
#         prediction_train = model(blurry_train.to(device))
#
#         plot_data_grid(originals_train, index_data, nRow, nCol, title='original size images, train')
#         plot_data_grid(blurry_train, index_data, nRow, nCol, title='input images, train')
#         plot_data_grid(prediction_train, index_data, nRow, nCol, title='output images, train')
#
#         nRow = 2
#         nCol = 1
#         originals_test_list = []
#         blurry_test_list = []
#         for idx in index_data:
#             originals_test, blurry_test = val_ds[idx]
#             originals_test = originals_test
#             blurry_test = blurry_test
#             originals_test_list.append(originals_test)
#             blurry_test_list.append(blurry_test)
#
#         originals_test = torch.tensor(np.stack(originals_test_list, 0))
#         blurry_test = torch.tensor(np.stack(blurry_test_list, 0))
#
#         prediction_test = model(blurry_test.unsqueeze(dim=1).to(device))
#
#         plot_data_grid(originals_test, index_data, nRow, nCol, title='original size images, test')
#         plot_data_grid(blurry_test, index_data, nRow, nCol, title='input images, test')
#         plot_data_grid(prediction_test, index_data, nRow, nCol, title='output images, test')

def inference(model):
    transform = transforms.Compose([
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Resize(1020)
    ])
    target_transform = transforms.Compose([
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Resize(2040)
    ])
    train_ds = DIV2KDataset(dir=DIV2K_PATH, transform=transform, target_transform=target_transform)
    val_ds = DIV2KDataset(dir=DIV2K_PATH, type='valid', transform=transform, target_transform=target_transform)

    train_dl = DataLoader(train_ds, batch_size=1, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=True)

    for image, label in train_dl:
        fig, ax = plt.subplots(1, 3)
        im = np.zeros((image.shape[-2], image.shape[-1], image.shape[-3]))
        im[:, :, 0] = image[0, 0, :, :]
        im[:, :, 1] = image[0, 1, :, :]
        im[:, :, 2] = image[0, 2, :, :]
        im = im / 255
        ax[0].imshow(im, vmin=0, vmax=1)
        im = np.zeros((label.shape[-2], label.shape[-1], label.shape[-3]))
        im[:, :, 0] = label[0, 0, :, :]
        im[:, :, 1] = label[0, 1, :, :]
        im[:, :, 2] = label[0, 2, :, :]
        im = im / 255
        ax[1].imshow(im, vmin=0, vmax=1)
        output = model(image.to(device)).to('cpu').detach().numpy()
        im = np.zeros((output.shape[-2], output.shape[-1], output.shape[-3]))
        im[:, :, 0] = output[0, 0, :, :]
        im[:, :, 1] = output[0, 1, :, :]
        im[:, :, 2] = output[0, 2, :, :]
        im -= im.min()
        im /= im.max()
        ax[2].imshow(im, vmin=0, vmax=1)
        plt.show()


if __name__ == '__main__':
    model = Unet().to(device)
    model.eval()
    model.load_state_dict(torch.load(f'./model.pth'))
    inference(model)
