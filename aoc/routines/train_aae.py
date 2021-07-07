import os

def reset_cuda_vis_dev():
    cuda_env_word = 'CUDA_VISIBLE_DEVICES'
    if cuda_env_word in os.environ:
    	cudas = os.environ[cuda_env_word].split(',')
    	cudas_line = ''
    	if len(cudas) > 1:
    	    for i in range(len(cudas)-1):
    	    	cudas_line += str(i) + ','
    	cudas_line += str(len(cudas)-1)
    	os.environ[cuda_env_word] = cudas_line
    	return len(cudas)
    else:
        return 4

#N_CUDAS = reset_cuda_vis_dev()
N_CUDAS = 1

import numpy as np
import torch
from torch.utils.data import Dataset
from aoc.model.pose.grids.grids import Grid
from aoc.model.pose.render import ObjectRenderer
from aoc.model.pose.datasets.dataset import RenderedDataset
from aoc.model.pose.datasets.augment import AAETransform
from aoc.model.pose.aae.aae import AEDataModule, AAE
from tqdm import tqdm
import os
from torchvision import transforms
import pytorch_lightning as pl
from aoc.model.classification.dataset import ManyObjectsRenderedDataset


def train_and_save_cans(workdir, gpu=1, rendered=True, epochs=30, mnames=None, latent_size=256):
    voc_path = '~/VOC'
    #voc_path = '/home/safoex/Documents/data/VOCtrainval_11-May-2012'
    # models_dir = '/home/safoex/Downloads/cat_food/models_fixed/'
    models_dir = '~/aaedata/models_fixed'
    models_names = mnames or ['tonno_low', 'pollo', 'polpa']
    # models_dir = 'scans'
    # models_names = ['fragola', 'tiramisu', 'pistacchi']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}
    # workdir = 'test_many_boxes'
    grider = Grid(3000, 20)
    ds = ManyObjectsRenderedDataset(grider, models,
                                    aae_render_tranform=AAETransform(0.5, voc_path, add_aug=True, add_patches=True))
    ds.set_mode('aae')
    if rendered:
        ds.load_dataset(workdir)
    else:
        ds.create_dataset(workdir)

    if N_CUDAS > 1 and gpu != 1:
    	trainer = pl.Trainer(gpus=gpu, max_epochs=epochs, distributed_backend='ddp')
    else:
        if N_CUDAS == 1:
            gpu = [gpu]
        trainer = pl.Trainer(gpus=gpu, max_epochs=epochs)

    dm = AEDataModule(ds, workdir, batch_size=64, num_workers=16)
    ae = AAE(128, latent_size, (128, 256, 256, 512))
	
    #ae.lr = ae.lr * (gpu if gpu > 0 else N_CUDAS)
    trainer.fit(ae, dm)
    torch.save(ae.state_dict(), workdir+ '/' + 'multi256.pth')

import sys
import time
if __name__ == "__main__":
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    start = time.time()
    mnames = None
    if len(sys.argv) > 5: 
        mnames = sys.argv[5:]
    NO_RENDER = "NO_RENDER"
    rendered = False
    if NO_RENDER in mnames:
        mnames.remove(NO_RENDER)
        rendered = True
    gpu = int(sys.argv[2])
    workdir_path = sys.argv[1]
    epochs = int(sys.argv[3])
    latent_size = int(sys.argv[4])
    train_and_save_cans(sys.argv[1],int(sys.argv[2]), rendered, int(sys.argv[3]), mnames, latent_size=latent_size)
    end = time.time()
    print('elapsed time is ' + str(end - start))
