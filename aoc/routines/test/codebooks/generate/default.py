from aoc.routines.test.settings import *
from aoc.model.pose.aae.aae import AAE, AEDataModule
from aoc.model.pose.datasets.dataset import OnlineRenderDataset, Grid
from aoc.model.pose.codebooks.codebook import *
import numpy as np
import os
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    wdir = workdir + '/datasets_augmented'

    ae = AAE(128, 128, (128, 256, 256, 512))
    ae.load_state_dict(torch.load(wdir + '/ae128.pth'))

    ae.cuda()

    grider = default_inference_grider
    ords = OnlineRenderDataset(grider, example_model_path)

    cdbk = Codebook(ae, ords)

    cdbk.save(wdir + '/cdbk_default.pt')
