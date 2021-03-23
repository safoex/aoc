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
    # wdir  = '/home/safoex/Documents/data/aae/babyfood3'
    wdir  = '/home/safoex/Documents/data/aae/release6/redboxes'
    # ae = AAE(128, 128, (128, 256, 256, 512))
    ae = AAE(128, 256, (128, 256, 256, 512))
    ae.load_state_dict(torch.load(wdir + '/multi256.pth'))

    ae.cuda()

    models_dir = '/home/safoex/Documents/data/aae/models/scans/'
    models_names = ['tiramisu', 'pistacchi', 'cioccolato', 'vaniglia', 'meltacchin']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}

    ords = OnlineRenderDataset(default_small_grider, models['tiramisu']['model_path'], directional_light_intensity=(3, 15))
    idx = np.random.randint(len(ords))
    render, _, _ = ords[idx]

    img = render[np.newaxis, :, :, :]
    print(render.shape)
    with torch.no_grad():
        reconst = ae.forward(torch.tensor(img).cuda())

    render = np.moveaxis(render, 0, 2)
    reconst = np.moveaxis(reconst.cpu().numpy()[0,:,:,:], 0, 2)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(render)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(reconst)
    plt.show()
