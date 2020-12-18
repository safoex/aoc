from mitmpose.routines.test.settings import *
from mitmpose.model.pose.aae.aae import AAE, AEDataModule
from mitmpose.model.pose.datasets.dataset import OnlineRenderDataset, Grid
import numpy as np
import os
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # wdir = workdir + '/datasets_augmented'
    wdir  = '/home/safoex/Documents/data/aae/babyfood3'
    # ae = AAE(128, 128, (128, 256, 256, 512))
    ae = AAE(128, 256, (128, 256, 256, 512))
    ae.load_state_dict(torch.load(wdir + '/multi256.pth'))

    ae.cuda()

    ords = OnlineRenderDataset(default_small_grider, example_model_path, directional_light_intensity=(3, 15))
    idx = np.random.randint(len(ords))
    render, _, _ = ords[idx]

    with torch.no_grad():
        reconst = ae.forward(render)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(render)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(reconst)
    plt.show()
