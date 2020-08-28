from mitmpose.model.so3.dataset import RenderedDataset
from mitmpose.model.so3.grids import in_plane_rot


class RenderedDataset2D(RenderedDataset):
    def __init__(self, model_path=None, size=500, res=128):
        super().__init__(model_path, in_plane_rot, size, res)
