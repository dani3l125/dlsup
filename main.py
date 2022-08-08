import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


def initialize_loss():
    """
    Initialize the loss variables
    :return: The VGG model, the activation dictionary, the initialized loss
    """
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()

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


vgg, activations, loss = initialize_loss()  # initialize the vgg


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
