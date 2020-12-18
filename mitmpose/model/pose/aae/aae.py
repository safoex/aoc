import torch

from mitmpose.model.pose.grids.grids import Grid
from mitmpose.model.pose.datasets.augment import AugmentedDataset, AAETransform

from mitmpose.model.pose.aae.ae import AE, AEDataModule
import pytorch_lightning as pl


class AAE(AE):
    def __init__(self, image_size=128, latent_size=128, filters=(128, 256, 256, 512), lae_inside=None):
        super().__init__(image_size, latent_size, filters, lae_inside)

    def training_step(self, batch, batch_idx):
        x, m, p, x_aug = batch
        x_hat = self.forward(x_aug)
        loss = self.loss_aae(x, x_hat, bootstrap_ratio=4)
        result = pl.TrainResult(loss)
        # result.log('loss', loss, prog_bar=True)
        return result


if __name__ == "__main__":
    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    t = AAETransform(0.5, '/home/safoex/Documents/data/VOCtrainval_11-May-2012')
    grider = Grid(500, 20)
    ds = AugmentedDataset(grider, fuze_path, transform=t)
    ds.create_dataset('test_save4')
    # ds.load_dataset('test_save4')

    ae = AAE(128, 128, (128, 256, 256, 512))
    # train_ae(ae, dataset, iters=3000, save_every=30, save_path='test_save4/recons.jpg', batch_size=128)
    dm = AEDataModule(ds, 'test_save4', batch_size=32)
    pl.callbacks.early_stopping.EarlyStopping()
    trainer = pl.Trainer(gpus=1, max_epochs=60, )
    # trainer.fit(ae, datamodule=dm)
    torch.save(ae.state_dict(), 'test_save4/ae128.pth')
