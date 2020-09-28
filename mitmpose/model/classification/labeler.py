from mitmpose.model.so3.aae import *
from mitmpose.model.so3.dataset import *
from mitmpose.model.classification.dataset import *
from mitmpose.model.classification.classifier import *


class Labeler:
    def __init__(self, ds_many, ae_small=(128, 64, (32, 64, 64, 128)), ae_big=(128, 128, (128, 256, 256, 512))):
        self.aae_small = AAE(*ae_small)
        self.aae_big_settings = ae_big
        self.ds_many = ds_many

    def train_small_ae(self):
        pass

