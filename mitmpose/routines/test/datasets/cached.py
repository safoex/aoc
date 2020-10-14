from mitmpose.model.pose.datasets.dataset import RenderedDataset
from mitmpose.routines.test.settings import *
import numpy as np
import os

if __name__ == '__main__':
    wdir = workdir + '/datasets_cached'
    if not os.path.exists(wdir):
        os.mkdir(wdir)

    rds = RenderedDataset(default_small_grider, example_model_path)

    rds.load_dataset(wdir)

    idx = np.random.randint(0, len(rds), 1)
    img, mask, _ = rds[idx]
    rds.test_show(img[0].numpy(), mask[0].numpy())
