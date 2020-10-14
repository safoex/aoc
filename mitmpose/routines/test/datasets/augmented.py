from mitmpose.model.pose.datasets.dataset import AugmentedAndRenderedDataset, Grid
from mitmpose.model.pose.datasets.augment import AAETransform
from mitmpose.routines.test.settings import *
import numpy as np
import os


if __name__ == '__main__':
    wdir = workdir + '/datasets_augmented'
    if not os.path.exists(wdir):
        os.mkdir(wdir)

    augmenter = AAETransform(bg_image_dataset_folder=voc_path)
    ards = AugmentedAndRenderedDataset(default_small_grider, example_model_path, augmenter)

    ards.load_dataset(wdir)

    idx = np.random.randint(0, len(ards), 1)
    img, mask, rot, img_aug = ards[idx]

    ards.reconstruction_dataset.test_show(img_aug[0].numpy(), mask[0].numpy())
