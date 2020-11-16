from mitmpose.model.pose.codebooks.codebook import *


class PoseGradient:
    def __init__(self, grider, model_path, ae):
        self.model = model_path
        self.grider = grider
        self.ae = ae


