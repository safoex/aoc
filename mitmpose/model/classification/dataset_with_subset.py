from mitmpose.model.classification.dataset import ManyObjectsRenderedDataset, Grid
import torch


class ManyObjectsDatasetWithSubset(ManyObjectsRenderedDataset):
    def __init__(self, grider: Grid, grid_subsets, models: dict, aae_render_tranform, classification_transform=None, res=128,
                 camera_dist=None,
                 render_res=640, intensity_render=10, intensity_augment=(2, 20), online=False, aae_scale_factor=1.2):

        super().__init__(grider, models, aae_render_tranform, classification_transform, res, camera_dist,
                         render_res, intensity_render, intensity_augment, online, aae_scale_factor=aae_scale_factor)

        self.grid_subsets = grid_subsets # binary masks
        if self.grid_subsets is not None:
            self.grid_subset_len = torch.sum(grid_subsets).item()
            self.indices = torch.flatten(torch.nonzero(torch.flatten(torch.tensor(self.grid_subsets)))).cpu()
            # print(self.indices)
        else:
            self.grid_subset_len = len(self.grider)

    def __len__(self):
        return self.grid_subset_len

    def __getitem__(self, idx):
        if self.grid_subsets is not None:
            # print(idx, len(self.indices))
            return super().__getitem__(self.indices[idx].item())
        else:
            return super().__getitem__(idx)
