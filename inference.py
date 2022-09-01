import matplotlib.pyplot as plt
import torch
from utils.Data import DIV2KDataset, plot_data_grid
from torchvision import transforms
from Models import Unet
from torch.utils.data import DataLoader
import numpy as np
import argparse
import yaml
import glob
import cv2
import onnx
import onnxruntime as ort

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='configuration file path')
parser.add_argument('--cfg', type=str, default='cfg_ferrum.yaml',
                    help='path to configuration file')

args = parser.parse_args()
with open(args.cfg, 'r') as stream:
    cfg = yaml.safe_load(stream)

DIV2K_PATH = cfg['DIV2K_PATH']

MODEL_PATH = cfg['MODEL_PATH']
NAME = cfg['NAME']


def plot_prepare(image):
    im = np.zeros((image.shape[-2], image.shape[-1], image.shape[-3]))
    im[:, :, 0] = image[0, 0, :, :]
    im[:, :, 1] = image[0, 1, :, :]
    im[:, :, 2] = image[0, 2, :, :]
    im -= im.min()
    im /= im.max()
    return im


def inference_prepare(image):
    im = np.zeros((1, 3, image.shape[0], image.shape[1]))
    im[0, 0, :, :] = image[:, :, 0]
    im[0, 1, :, :] = image[:, :, 1]
    im[0, 2, :, :] = image[:, :, 2]
    return im


def inference(model, data='val'):
    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    target_transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    ds = DIV2KDataset(dir=DIV2K_PATH, transform=transform, target_transform=target_transform) if \
        data == 'train' else DIV2KDataset(dir=DIV2K_PATH, type='valid', transform=transform,
                                          target_transform=target_transform)

    dl = DataLoader(ds, batch_size=1, num_workers=4, pin_memory=True)

    ds_plot = DIV2KDataset(dir=DIV2K_PATH) if data == 'train' else \
        DIV2KDataset(dir=DIV2K_PATH, type='valid', )

    dl_plot = DataLoader(ds, batch_size=1, num_workers=4, pin_memory=True)

    for (image, label), (image_plot, label_plot) in zip(dl, dl_plot):
        fig, ax = plt.subplots(1, 3)
        im = plot_prepare(image)

        ax[0].imshow(im, vmin=0, vmax=1)
        im = plot_prepare(label_plot)
        ax[1].imshow(im, vmin=0, vmax=1)

        output = model(image.to(device)).to('cpu').detach().numpy()
        im = plot_prepare(output)
        ax[2].imshow(im, vmin=0, vmax=1)
        plt.show()


def video_inference(model, path='/home/daniel/dlsup/sample.mp4'):
    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    image = inference_prepare(image).astype(np.float32)
    im = plot_prepare(image)
    torch.onnx.export(model,  # model being run
                      torch.tensor(image).to(device),  # model input (or a tuple for multiple inputs)
                      "dlsup.onnx",  # where to save the model (can be a file or file-like object)
                      input_names=['input'],  # the model's input names
                      output_names=['output'])  # the model's output names
    onnx_model = onnx.load("dlsup.onnx")
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession('dlsup.onnx',
                                    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                               'CPUExecutionProvider'])

    count = 0
    while success and count < 10:
        fig, ax = plt.subplots(1, 2)
        output = ort_sess.run(None, {'input': transform(torch.tensor(image).to('cpu')).numpy()})[0]
        output_im = plot_prepare(output)
        # cv2.imshow('Output', output_im)
        ax[0].imshow(im, vmin=0, vmax=1)
        ax[1].imshow(output_im, vmin=0, vmax=1)
        cv2.imwrite("newvideo/frame%d.jpg" % count, output_im * 255)  # save frame as JPEG file
        # print('Read a new frame: ', success)
        # plt.show()
        success, image = vidcap.read()
        if not success:
            break
        image = inference_prepare(image).astype(np.float32)
        im = plot_prepare(image)
        count += 1


def save_video():
    img_array = []
    count = 0
    for filename in glob.glob('newvideo/*.jpg'):
        filename = "newvideo/frame" + str(count) + ".jpg"
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
        count += 1
    # out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))
    out = cv2.VideoWriter('inference_sample.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, size)
    # out = cv2.VideoWriter()

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':
    model = Unet().to(device)
    model.eval()
    model.load_state_dict(torch.load(f'/home/daniel/exp1_last.pth'))
    video_inference(model)
    save_video()
