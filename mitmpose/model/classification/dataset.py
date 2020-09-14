import numpy as np
import torch
from torch.utils.data import Dataset
from mitmpose.model.so3.grids import Grid
from mitmpose.model.so3 import ObjectRenderer
from tqdm import tqdm
import os


class ManyObjectRenderedDataset(Dataset):
    def __init__(self, grider: Grid, models: dict, res=128, camera_dist=0.5, render_res=640):
