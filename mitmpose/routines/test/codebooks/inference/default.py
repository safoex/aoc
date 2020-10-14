from mitmpose.routines.test.settings import *
from mitmpose.model.pose.aae.aae import AAE, AEDataModule
from mitmpose.model.pose.datasets.dataset import OnlineRenderDataset, Grid
from mitmpose.model.pose.codebooks.codebook import *
import numpy as np
import os
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group


if __name__ == '__main__':
    wdir = workdir + '/datasets_augmented'

    ae = AAE(128, 128, (128, 256, 256, 512))
    ae.load_state_dict(torch.load(wdir + '/ae128.pth'))

    ae.cuda()

    grider = default_inference_grider
    ords = OnlineRenderDataset(grider, example_model_path, directional_light_intensity=(3, 15))

    cdbk = Codebook(ae, ords)

    cdbk.load(wdir + '/cdbk_default.pt')

    rots = special_ortho_group.rvs(3, 100)

    rots_cdbk = [grider.grid[cdbk.best(cdbk.latent(ords.objren.render_and_crop(rot)[0]))] for rot in rots]

    dif = [rot.T * rot_cdbk for rot, rot_cdbk in zip(rots, rots_cdbk)]

    norms = np.linalg.norm(Rotation.from_matrix(dif).as_euler('xyz'), axis=1)

    print(np.mean(norms), np.min(norms), np.max(norms))





