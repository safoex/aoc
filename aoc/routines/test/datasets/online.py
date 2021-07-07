from aoc.model.pose.datasets.dataset import OnlineRenderDataset, Grid
from aoc.routines.test.settings import *
import numpy as np


if __name__ == '__main__':
    ords = OnlineRenderDataset(default_small_grider, example_model_path)

    img, mask = None, None
    for idx in np.random.randint(0, len(ords), 10):
        img, mask, _ = ords[idx]

    ords.test_show(img, mask)

