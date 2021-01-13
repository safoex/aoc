from torch.utils.data import Dataset
from mitmpose.model.pose.grids.grids import Grid
from mitmpose.model.pose.datasets.augment import AAETransform
from mitmpose.model.pose.datasets.dataset import OnlineRenderDataset, RenderedDataset, AugmentedAndRenderedDataset
from mitmpose.model.pose.aae.aae import AEDataModule, AAE
from torchvision import transforms
import pytorch_lightning as pl


class ManyObjectsRenderedDataset(Dataset):
    default_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(236),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([
            transforms.Resize(236),
            transforms.RandomAffine(20, (0.1, 0.1), (0.9, 1.1)),
            transforms.RandomCrop(224),
        ], 0.5),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_inference = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, grider: Grid, models: dict, aae_render_tranform, classification_transform=None, res=128, camera_dist=None,
                 render_res=640, intensity_render=10, intensity_augment=(2, 20), online=False, aae_scale_factor=1.2):
        self.grider = grider
        self.models = models
        self._datasets = None
        self.default_params = {
            'camera_dist': camera_dist,
            'render_res': render_res,
            'res': res,
            'intensity_reconstruction': intensity_render,
            'intensity_augment': intensity_augment,
            'augmenter': aae_render_tranform,
            'aae_scale_factor': aae_scale_factor
        }
        self.labels = {}
        self.transform = classification_transform or self.default_transform
        self.aae_render_transform = aae_render_tranform
        self.mode = 'class'
        self.class_ds = AugmentedAndRenderedDataset

        self.online = online
        if self.online:
            self.default_params['aug_class'] = OnlineRenderDataset


    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = {}
            label = 0
            for model_name, params in self.models.items():
                mparams = self.default_params.copy()
                mparams.update(params)
                self._datasets[model_name] = self.class_ds(self.grider, **mparams)
                self.labels[model_name] = label
                label += 1
        return self._datasets

    def set_mode(self, mode):
        self.mode = mode

    def create_dataset(self, folder):
        for model_name, params in self.models.items():
            self.datasets[model_name].create_dataset(folder + '/' + model_name)

    def load_dataset(self, folder):
        for model_name, dataset in self.datasets.items():
            dataset.load_dataset(folder + '/' + model_name)

    def __len__(self):
        return sum(len(ds) for ds in self.datasets.values())

    def __getitem__(self, idx):
        obj = None
        for obj, ds in self.datasets.items():
            if idx >= len(ds):
                idx -= len(ds)
            else:
                break

        assert obj is not None

        if self.mode == 'class':
            ds_idx = -1
            if self.online:
                ds_idx = 0
            return self.transform(self.datasets[obj][idx][ds_idx]), self.labels[obj]
        else:
            return self.datasets[obj][idx]


if __name__ == '__main__':
    models_dir = '/home/safoex/Downloads/cat_food/models_fixed/'
    models_names = ['tonno_low', 'pollo', 'polpa']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': 140} for mname in models_names}
    workdir = 'test_many_reconstr'
    grider = Grid(30, 10)
    ds = ManyObjectsRenderedDataset(grider, models,
                                    aae_render_tranform=AAETransform(0.5, '/home/safoex/Documents/data/VOCtrainval_11-May-2012', add_aug=False))
    ds.set_mode('aae')
    ds.create_dataset(workdir)

    trainer = pl.Trainer(gpus=1, max_epochs=100)

    dm = AEDataModule(ds, workdir)
    ae = AAE(128, 64, (32, 64, 64, 128))

    trainer.fit(ae, dm)

