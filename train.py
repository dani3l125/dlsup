import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.Data import DIV2KDataset, plot_data_grid
from utils.metrics import compute_accuracy, compute_loss
from torchvision import transforms
from Models import Unet
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DIV2K_PATH = '.'
DEFAULT_LR = 0.001
DEFAULT_BS = 1
EPOCHS = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# plot train and test metric along epochs
def plot_curve_error(train_mean, train_std, test_mean, test_std, x_label, y_label, title, identity=[]):
    plt.figure(figsize=(10, 8))
    plt.title(title)

    alpha = 0.1

    plt.plot(range(len(train_mean)), train_mean, '-', color='red', label='train')
    plt.fill_between(range(len(train_mean)), train_mean - train_std, train_mean + train_std, facecolor='red',
                     alpha=alpha)

    plt.plot(range(len(test_mean)), test_mean, '-', color='blue', label='test')
    plt.fill_between(range(len(test_mean)), test_mean - test_std, test_mean + test_std, facecolor='blue', alpha=alpha)

    if not identity == []:
        plt.plot(range(len(identity)), identity, '--', color='green', label='identity')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{title}.png')

def train(model, visualize_data=False):

    optimizer = Adam(model.parameters(), lr=DEFAULT_LR)
    lr_scheduler = ReduceLROnPlateau(optimizer)
    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #transforms.Resize(1020)
    ])
    target_transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #transforms.Resize(2040)
    ])
    train_ds = DIV2KDataset(dir=DIV2K_PATH, transform=transform, target_transform=target_transform)
    val_ds = DIV2KDataset(dir=DIV2K_PATH, type='valid', transform=transform, target_transform=target_transform)
    # test_ds = DIV2KDataset(dir=DIV2K_PATH, type='test', transform=transform, target_transform=target_transform)

    train_dl = DataLoader(train_ds, batch_size=DEFAULT_BS, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=DEFAULT_BS, num_workers=4, pin_memory=True)
    # test_dl = DataLoader(test_ds, batch_size=DEFAULT_BS, num_workers=4, pin_memory=True)

    loss_mean_train = np.zeros(EPOCHS)
    loss_std_train = np.zeros(EPOCHS)
    psnr_mean_train = np.zeros(EPOCHS)
    psnr_std_train = np.zeros(EPOCHS)
    ssim_mean_train = np.zeros(EPOCHS)
    ssim_std_train = np.zeros(EPOCHS)

    loss_mean_val = np.zeros(EPOCHS)
    loss_std_val = np.zeros(EPOCHS)
    psnr_mean_val = np.zeros(EPOCHS)
    psnr_std_val = np.zeros(EPOCHS)
    ssim_mean_val = np.zeros(EPOCHS)
    ssim_std_val = np.zeros(EPOCHS)

    def train_epoch():
        loss_epoch = []
        psnr_epoch = []
        ssim_epoch = []

        model.train()

        for index_batch, (im, target) in enumerate(train_dl):
            im = im.to(device)
            target = target.to(device)
            # prediction
            prediction = model(im)

            # loss - modeified according to psnr function
            loss = compute_loss(prediction, target)

            # accuracy
            psnr, ssim = compute_accuracy(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.item())
            psnr_epoch.append(psnr)
            ssim_epoch.append(ssim)
            print(f'batch:{index_batch} | loss:{loss} | psnr:{psnr} | ssim:{ssim}')

        loss_mean_epoch = np.mean(loss_epoch)
        loss_std_epoch = np.std(loss_epoch)

        psnr_mean_epoch = np.mean(psnr_epoch)
        psnr_std_epoch = np.std(psnr_epoch)
        ssim_mean_epoch = np.mean(ssim_epoch)
        ssim_std_epoch = np.std(ssim_epoch)

        loss = {'mean': loss_mean_epoch, 'std': loss_std_epoch}
        psnr = {'mean': psnr_mean_epoch, 'std': psnr_std_epoch}
        ssim = {'mean': ssim_mean_epoch, 'std': ssim_std_epoch}

        return (loss, psnr, ssim)

    def valid_epoch(epoch_i, return_input=False, test=False):
        loss_epoch = []
        psnr_epoch = []
        ssim_epoch = []

        model.eval()
        dataloader = val_dl
        # dataloader = val_dl if not test else test_dl

        for index_batch, (im, target) in enumerate(dataloader):

            im = im.to(device)
            target = target.to(device)
            # prediction
            prediction = model(im)

            # loss - modeified according to psnr function
            loss = compute_loss(prediction, target)

            # accuracy
            psnr, ssim = compute_accuracy(prediction, target)

            if return_input:
                # input accuracy
                psnr, ssim = compute_accuracy(target, im)

            loss_epoch.append(loss.item())
            psnr_epoch.append(psnr)
            ssim_epoch.append(ssim)

        loss_mean_epoch = np.mean(loss_epoch)
        loss_std_epoch = np.std(loss_epoch)

        psnr_mean_epoch = np.mean(psnr_epoch)
        psnr_std_epoch = np.std(psnr_epoch)
        ssim_mean_epoch = np.mean(ssim_epoch)
        ssim_std_epoch = np.std(ssim_epoch)

        if epoch_i > 10:
            lr_scheduler.step(loss.item())

        loss = {'mean': loss_mean_epoch, 'std': loss_std_epoch}
        psnr = {'mean': psnr_mean_epoch, 'std': psnr_std_epoch}
        ssim = {'mean': ssim_mean_epoch, 'std': ssim_std_epoch}

        return (loss, psnr, ssim)

    if visualize_data:
        nRow = 1
        nCol = 10
        index_data = np.arange(nRow * nCol)  # show only first images

        for index_batch, (im, target) in enumerate(train_dl):
            plot_data_grid(im, index_data, nRow, nCol ,'sample from augmented batch')
            plot_data_grid(target, index_data, nRow, nCol)

    # ================================================================================
    #
    # iterations for epochs
    #
    # ================================================================================
    for i in tqdm(range(EPOCHS)):
        # ================================================================================
        # training
        # ================================================================================
        (loss_train, psnr_train, ssim_train) = train_epoch()
        print(f'Train Epoch: {i} | Loss: {loss_train} | PSNR: {psnr_train} | SSIM: {ssim_train}')

        loss_mean_train[i] = loss_train['mean']
        loss_std_train[i] = loss_train['std']
        psnr_mean_train[i] = psnr_train['mean']
        psnr_std_train[i] = psnr_train['std']
        ssim_mean_train[i] = ssim_train['mean']
        ssim_std_train[i] = ssim_train['std']

        # ================================================================================
        # testing (validation)
        # ================================================================================
        (loss_test, psnr_test, ssim_test) = valid_epoch(i)
        print(f'Validation Epoch: {i} | Loss: {loss_test} | PSNR: {psnr_test} | SSIM: {ssim_test}')

        loss_mean_val[i] = loss_test['mean']
        loss_std_val[i] = loss_test['std']
        psnr_mean_val[i] = psnr_test['mean']
        psnr_std_val[i] = psnr_test['std']
        ssim_mean_val[i] = ssim_test['mean']
        ssim_std_val[i] = ssim_test['std']

    if visualize_data:
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

    # loss
    plot_curve_error(loss_mean_train, loss_std_train, loss_mean_val, loss_std_val, 'epoch', 'losses', 'LOSS')
    # accuracy - PSNR
    plot_curve_error(psnr_mean_train, psnr_std_train, psnr_mean_val, psnr_std_val, 'epoch', 'accuracy', 'PSNR')
    # accuracy - SSIM
    plot_curve_error(ssim_mean_train, ssim_std_train, ssim_mean_val, ssim_std_val, 'epoch', 'accuracy', 'SSIM')

    # notice that the 'train' signals were computed each batch while the test signals are computed at the end of the epoch
    torch.save(model.state_dict(), './model.pth')

    (loss_test, psnr_test, ssim_test) = valid_epoch(0, test=True)
    print('Test PSNR: ', psnr_test['mean'])
    print('Test SSIM: ', ssim_test['mean'])


if __name__ == '__main__':
    model = Unet().to(device)
    train(model)
