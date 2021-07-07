import os


def reset_cuda_vis_dev():
    cuda_env_word = 'CUDA_VISIBLE_DEVICES'
    if cuda_env_word in os.environ:
        cudas = os.environ[cuda_env_word].split(',')
        cudas_line = ''
        if len(cudas) > 1:
            for i in range(len(cudas)-1):
                cudas_line += str(i) + ','
        cudas_line += str(len(cudas)-1)
        os.environ[cuda_env_word] = cudas_line
        return len(cudas)
    else:
        return 4

#N_CUDAS = reset_cuda_vis_dev()
N_CUDAS = 1

from aoc.model.classification.labeler_ambigous import *
import sys


def label(workdir, gpu, mnames, latent_size, grad_steps, exp_shrink):
    # models_dir = '/home/safoex/Downloads/cat_food/models_fixed/'
    models_names = mnames or ['tonno_low', 'pollo', 'polpa']
    wdir_root = '~/aaedata'
    models_dir =  wdir_root + '/models'
    # models_names = ['fragola', 'pistacchi', 'tiramisu']
    workdir = workdir or wdir_root + '/cans'
    ae_path = workdir + '/multi256.pth'
    

    models = {mname: {'model_path': models_dir + '/' + mname + '.obj'} for mname in models_names}
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    
    N = 100
    grider = Grid(N, 1)

    ae = AAE(128, latent_size, (128, 256, 256, 512))
    ae.load_state_dict(torch.load(ae_path))

    device = torch.device('cuda:%d' % gpu)
    ae.to(device)

    labeler = AmbigousObjectsLabeler(models, grider_label=grider, grider_codebook=Grid(4000, 40), ae=ae, magic_grad_steps=grad_steps, magic_shrink_exp=exp_shrink)
    labeler.save_codebooks(workdir)
    labeler.load_codebooks(workdir)
    # labeler.load(workdir)
    # labeler.recalculate_fin_labels()
    labeler.save(workdir)

    n_models = len(models_names)
    print(models_names)
    for i_from, i_to in itertools.product(range(n_models), range(n_models)):
        if i_from != i_to:
            top_n = N
            wdir_save = workdir + '/' + 'tops_%d_%d' % (i_from, i_to)
            labeler.knn_median = 7

            print_out_sim_views(labeler, models_names, models, i_from, i_to, top_n, wdir_save)


if __name__ == "__main__":
    workdir = sys.argv[1]
    gpu = int(sys.argv[2])
    latent_size = int(sys.argv[3])
    grad_steps = int(sys.argv[4])
    exp_shrink = float(sys.argv[5])
    mnames = sys.argv[6:]
    label(workdir, gpu, mnames, latent_size, grad_steps, exp_shrink)
