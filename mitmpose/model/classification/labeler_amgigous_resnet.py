from mitmpose.model.classification.dataset import *
from mitmpose.model.pose.datasets.dataset import OnlineRenderDataset
import torch
import itertools
from tqdm.auto import trange, tqdm
from torchvision import models
from torch import nn

class AmbigousObjectsLabelerResnet:
    def __init__(self, models, grider, resnet):
        self.models = models
        self.grider = grider
        self.resnet = resnet
        self._codebooks = None
        self._simililarities = None
        self.model_list = [m for m in self.models]
        self.model_idx = {m: i for i,m in enumerate(self.model_list)}
        self._labels = None

    @property
    def codebooks(self):
        if self._codebooks is None:
            self.resnet.eval()
            self._codebooks = {}
            for mname, mprop in self.models.items():
                ds = OnlineRenderDataset(self.grider, mprop['model_path'], camera_dist=None)
                self._codebooks[mname] = torch.zeros((len(self.grider.grid), ))
                self._codebooks[mname].codebook

        return self._codebooks





if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    model.freeze()
    features = nn.Sequential(*(list(model.children())[:-1]))
