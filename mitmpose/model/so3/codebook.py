from mitmpose.model.so3.aae import AAE
from mitmpose.model.so3.dataset import RenderedDataset, IndexedDataset
from mitmpose.model.so3.grids import Grid
from torch.utils.data import DataLoader
import torch

from mitmpose.model.so3.render import ObjectRenderer
from scipy.stats import special_ortho_group
import numpy as np


class Codebook:
    def __init__(self, model: AAE, dataset: RenderedDataset, batch_size=32):
        self.model = model
        self.dataset = IndexedDataset(dataset)
        self._codebook = None
        self.batch_size = batch_size

    @property
    def codebook(self):
        if self._codebook is None:
            dl = DataLoader(self.dataset, batch_size=self.batch_size)
            model_device = next(self.model.parameters()).device
            self._codebook = torch.zeros((len(self.dataset), self.model.latent_size), dtype=torch.float32).to(device=model_device)
            self.model.eval()
            for imgs, _, _, idcs in dl:
                with torch.no_grad():
                    self._codebook[idcs, :] = self.model.encoder.forward(imgs.cuda())


            self.normalize_(self._codebook)

        return self._codebook

    @staticmethod
    def normalize(code):
        if len(code.shape) > 1:
            norm = code.norm(p=2, dim=1, keepdim=True)
            return code.div(norm)
        else:
            return code / torch.norm(code)

    @staticmethod
    def normalize_(code):
        if len(code.shape) > 1:
            norm = code.norm(p=2, dim=1, keepdim=True)
            code.div_(norm)
        else:
            code.div_(torch.norm(code))

    def latent(self, img, norm=True):
        img = np.moveaxis(img, 2, 0) / 255.0
        img = img[np.newaxis, :, :, :]
        self.model.eval()
        code = self.model.encoder.forward(torch.Tensor(img).cuda())
        if norm:
            code /= torch.norm(code)
        return code

    def cos_sim(self, codes1, codes2=None, norm1=False, norm2=False):
        if codes2 is None:
            codes2 = self.codebook
            norm2 = True

        if not norm1:
            self.normalize_(codes1)

        if not norm2:
            self.normalize_(codes2)

        return torch.sum(codes1 * codes2, dim=1)

    def best(self, code, norm=True):
        if not norm:
            self.normalize_(code)

        return torch.argmax(self.cos_sim(code)).item()
        # self.dataset.dataset.grider.grid[]




if __name__ == "__main__":
    ae = AAE(128, 32, (8, 16, 16, 32)).cuda()
    ae.load_state_dict(torch.load('test_save4/ae32.pth'))

    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    grider = Grid(500, 20)
    ds = RenderedDataset(grider, fuze_path)
    ds.load_dataset('test_save4')

    codebook = Codebook(ae, ds)

    objren = ObjectRenderer(fuze_path)
    rot = special_ortho_group.rvs(3)
    img, depth = objren.render_and_crop(rot)
    lat = codebook.latent(img)
    print(ds.grider.grid[codebook.best(lat)])
    print(rot)
