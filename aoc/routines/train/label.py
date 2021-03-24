from mitmpose.model.classification.labeler_ambigous import *
import sys


if __name__ == '__main__':
    # models_dir = '/home/safoex/Downloads/cat_food/models_fixed/'
    # models_names = ['tonno_low', 'pollo', 'polpa']
    wdir_root = '/home/safoex/Documents/data/aae'
    models_dir =  wdir_root + '/models/scans'
    models_names = ['fragola', 'pistacchi', 'tiramisu']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj'} for mname in models_names}
    workdir = wdir_root + '/test_labeler'
    if not os.path.exists(workdir):
        os.mkdir(workdir)

    grider = Grid(300, 1)

    ae = AAE(128, 256, (128, 256, 256, 512))
    ae_path = wdir_root + '/multi_boxes256.pth'
    ae.load_state_dict(torch.load(ae_path))

    ae.cuda()

    labeler = AmbigousObjectsLabeler(models, grider_label=grider, grider_codebook=Grid(1000, 10), ae=ae)
    # labeler.load_codebooks(workdir)
    # labeler.load(workdir)
    labeler.save(workdir)

    n_models = len(models_names)
    for i_from, i_to in itertools.product(range(n_models), range(n_models)):
        if i_from != i_to:
            top_n = 300
            wdir_save = workdir + '/' + 'tops_%d_%d' % (i_from, i_to)
            labeler.knn_median = 10

            print_out_sim_views(labeler, i_from, i_to, top_n, wdir_save)
