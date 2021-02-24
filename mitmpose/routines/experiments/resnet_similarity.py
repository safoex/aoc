from mitmpose.routines.experiments.aae_labeled_baseline import *


if __name__ == "__main__":
    wdir_root = '/home/safoex/Documents/data/aae'
    models_dir = wdir_root + '/models/scans'
    # models_names = ['humana1', 'humana2']
    models_names = ['melpollo', 'meltacchin']
    workdir = wdir_root + '/babyfood2'
    ae_path = workdir + '/multi256.pth'
    N = 300
    abe = AAEBaselineExperiment(models_dir, models_names, workdir, ae_path, n_grid=N)

    # abe.plot_solo("aae solo")

    cl = torchvision.models.resnet50(pretrained=True).cuda()
    lat_size = cl.fc.in_features
    cl = torch.nn.Sequential(*list(cl.children())[:-1], torch.nn.Flatten())


    labeler = AmbigousObjectsLabeler(abe.models, grider_label=abe.grider, grider_codebook=Grid(1000, 10), ae=abe.ae,
                                     extra_encoder=cl, extra_latent_size=lat_size, extra_encoder_weight=1, ae_weight=0)

    labeler.load(workdir)
    labeler.recalculate_extra_cl_sims()
    labeler.recalculate_fin_labels()
    # print(labeler._extra_cl_sims)
    # labeler.save(workdir)

    i_from = 0
    i_to = 1
    resnet_loss = labeler._extra_cl_sims[:, i_from, i_to].cpu().numpy()[abe.get_indices(i_from,i_to)]


    # print(resnet_loss)

    abe.plot_comparison(resnet_loss, "ResNet based", "babyfood")