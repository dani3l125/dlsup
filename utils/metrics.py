import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.nn import MSELoss

def compute_accuracy(prediction, label):
    # # ssim

    # def eval_step(engine, batch):
    #     return batch

    # default_evaluator = Engine(eval_step)
    # metric = SSIM(data_range=1.0)
    # metric.attach(default_evaluator, 'ssim')
    # state = default_evaluator.run([[prediction, label]])
    # ssim = state.metrics['ssim']

    preds = prediction.squeeze().detach().cpu().numpy()
    targets = label.squeeze().detach().cpu().numpy()
    ssims = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        ssims[i] = structural_similarity(preds[i], targets[i], data_range=1)

    ssim = ssims.mean()

    # psnr

    prediction = prediction.squeeze(axis=1)
    label = label.squeeze(axis=1)
    mse_loss = torch.mean((prediction - label) ** 2)

    if mse_loss == 0.0:
        psnr = 100
    else:
        psnr = 10 * torch.log10(1 / mse_loss)

    psnr = psnr.item()

    return psnr, ssim

def get_loss_function():
    return MSELoss()