import numpy as np
from skimage.metrics import structural_similarity
from torch.nn import MSELoss
import torch
import torch.nn as nn
from torchvision.models import vgg16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOSS_HYPERPARAMETERS = {'features_loss': {'weight': 1,
                                          'j': 10
                                          },
                        'pixel_loss': {'weight': 1,
                                       },
                        'style_loss': {'weight': 1,
                                       'j_list': [4, 8, 12]
                                       }
                        }

from_file = True
MODEL_PATH = '/dlsup/utils/vgg.pth'

def initialize_loss():
    """
    Initialize the loss variables
    :return: The VGG model, the activation dictionary, the initialized loss
    """
    if from_file:
        vgg = vgg16().eval().to(device)
        vgg.load_state_dict(torch.load(MODEL_PATH))
    else:
        from torchvision.models import VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval().to(device)
        torch.save(vgg.state_dict(), '/home/daniel/vgg.pth')

    activation_indices = []
    for i, layer in enumerate(vgg.features):
        if isinstance(layer, nn.ReLU):
            activation_indices.append(i)

    activation = {}

    def get_hook_activation(i):
        def hook(model, input, output):
            activation[i] = output.detach()

        return hook

    for i in range(len(activation_indices)):
        vgg.features[activation_indices[i]].register_forward_hook(get_hook_activation(i))

    return vgg, activation, nn.MSELoss(reduction='sum')

VGG, ACTIVATION, LOSS = initialize_loss()  # initialize the vgg

def get_activation(output, label, vgg, activation, j):
    """
    :param output: The output of the network
    :param label: The target image
    :param vgg: The VGG model
    :param activation: The activation dictionary
    :param j: The number of activation
    :return: The outputs of the j-th layer of the VGG
    """
    vgg(output)
    phi_output = activation[j]
    vgg(label)
    phi_label = activation[j]
    return phi_output, phi_label


def features_loss(phi_output, phi_label, loss):
    """
    :param phi_output: The outputs of the j-th layer of the VGG with the output of the network as the input
    :param phi_label: The outputs of the j-th layer of the VGG with the target image as the input
    :param loss: The initialized loss
    :return: The feature loss with respect to the j-th layer of hte VGG
    """
    return loss(phi_output, phi_label) / torch.prod(torch.tensor(phi_output.shape[1:]))


def style_loss(phi_output, phi_label, loss):
    """
    :param phi_output: The outputs of the j-th layer of the VGG with the output of the network as the input
    :param phi_label: The outputs of the j-th layer of the VGG with the target image as the input
    :param loss: The initialized loss
    :return: The style loss with respect to the j-th layer of hte VGG
    """
    psi_output = phi_output.reshape((phi_output.shape[0], phi_output.shape[1], -1))
    psi_label = phi_label.reshape((phi_label.shape[0], phi_label.shape[1], -1))
    g_output = torch.einsum('bik,bjk->bij', psi_output, psi_output) / torch.prod(torch.tensor(phi_output.shape[1:]))
    g_label = torch.einsum('bik,bjk->bij', psi_label, psi_label) / torch.prod(torch.tensor(phi_label.shape[1:]))
    return loss(g_output, g_label)


def pixel_loss(output, label, loss):
    """
    :param output: The output of the network
    :param label: The target image
    :param loss: The initialized loss
    :return: The pixel wise loss between the images
    """
    return loss(output, label) / torch.prod(torch.tensor(output.shape[1:]))


def compute_loss(output, label, hyper_parameters=LOSS_HYPERPARAMETERS):
    loss = hyper_parameters['pixel_loss']['weight'] * pixel_loss(output, label, LOSS)
    features_phi_output, features_phi_label = get_activation(output,
                                                             label,
                                                             VGG,
                                                             ACTIVATION,
                                                             hyper_parameters['features_loss']['j'])
    loss += hyper_parameters['features_loss']['weight'] * features_loss(features_phi_output,
                                                                        features_phi_label,
                                                                        LOSS)
    j_list = hyper_parameters['style_loss']['j_list']
    for j in j_list:
        style_phi_output, style_phi_label = get_activation(output,
                                                           label,
                                                           VGG,
                                                           ACTIVATION,
                                                           j)
        loss += hyper_parameters['style_loss']['weight'] / len(hyper_parameters['style_loss']['j_list']) * style_loss(
            style_phi_output,
            style_phi_output,
            LOSS)
    return loss


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
        ssims[i] = structural_similarity(preds[i], targets[i], data_range=1, channel_axis=0)

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

if __name__ == '__main__':
        print('saved')
