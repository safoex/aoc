from aoc.model.classification.classifier_ambigous import *
from torchvision import transforms
from aoc.model.pose.aae.aae import AAE
from aoc.routines.test.settings import *

from matplotlib import pyplot as plt


if __name__ == '__main__':
    workdir = '/home/safoex/Documents/data/aae/babyfood3'
    # models_dir = '/content/drive/My Drive/iit/research/scans/models_fixed'
    # models_dir = workdir
    # models_names = ['fragola', 'pistacchi', 'tiramisu']
    models_names = ['meltacchin', 'melpollo']
    # models_names = ['humana1', 'humana2']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}

    # ae_path = workdir + '/multi_boxes256.pth'
    ae_path = workdir + '/multi256.pth'

    grider = Grid(200, 5)
    # grider_codebook = Grid(1000, 10)
    grider_codebook = Grid(4000, 40)

    ae = AAE(128, 256, (128, 256, 256, 512))
    ae.cuda()
    ae.eval()
    # ae_path = '/home/safoex/Documents/data/aae/cans_pth2/multi128.pth'
    ae.load_state_dict(torch.load(ae_path))
    cltrans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(236),
        transforms.RandomCrop(224),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds_args = {
        'grider': grider,
        'grider_codebook': grider_codebook,
        'models':models,
        'joint_ae': ae,
        'aae_render_tranform': AAETransform(0.5, voc_path, add_aug=False, add_patches=False),
        'classification_transform': cltrans
    }

    sue = SortedUncertaintyExperiment(ds_args, workdir, epochs=5, fraction_step=0.05, freeze_conv=True)
    sue.create_dataset()
    results = sue.series_experiment()
    fraction_midpoint = []
    corrects_left = []
    corrects_right = []
    for exp in results:
        begin = exp['begin'][0]
        end = exp['end'][0]
        fr = exp['fraction']
        fraction_midpoint.append((fr[0] + fr[1])/2)
        corrects_left.append(begin['corrects'])
        corrects_right.append(end['corrects'])

    plt.scatter(fraction_midpoint, corrects_left)
    plt.scatter(fraction_midpoint, corrects_right)
    plt.show()