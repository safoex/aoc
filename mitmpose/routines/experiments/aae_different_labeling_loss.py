from mitmpose.routines.experiments.aae_labeled_baseline import *


if __name__ == "__main__":
    wdir_root = '/home/safoex/Documents/data/aae'
    models_dir = wdir_root + '/models/scans'
    # models_names = ['humana1', 'humana2']
    # models_names = ['melpollo', 'meltacchin']
    models_names = ['mustardblue', 'mustardgreen']
    # workdir = wdir_root + '/babymilk2'
    workdir = wdir_root + '/mustard1/mustard'
    workdir_bad = wdir_root + '/mustard'
    ae_path = workdir + '/multi256.pth'
    N = 200
    abe = AAEBaselineExperiment(models_dir, models_names, workdir, ae_path, N)
    abe_bad = AAEBaselineExperiment(models_dir, models_names, workdir_bad, ae_path, 100)
    abe_bad7 = AAEBaselineExperiment(models_dir, models_names, wdir_root + '/mustard7', ae_path, 200)
    abe_bad13 = AAEBaselineExperiment(models_dir, models_names, wdir_root + '/mustard13', ae_path, 200)

    abe_bad_loss = np.zeros_like(abe.labeler_loss)
    abe_bad_loss[0::2] = abe_bad.labeler_loss
    abe_bad_loss[1::2] = abe_bad.labeler_loss
    # abe.plot_comparison(abe_bad_loss, "AAE with less gradient steps", "aae_bad")
    plt.figure(figsize=(4.5,4.5))
    title = "Different number of descent steps"
    plt.xlim([55, 165])
    plt.ylim([0.9965, 1.0005])
    abe.plot_label_pattern = "%s"
    abe.plot_one_of_many(abe_bad_loss, "10 steps", title, try_plot_self=False)
    # abe.plot_one_of_many(abe_bad7.labeler_loss, "7 steps", title, self_label="20 steps")
    abe.plot_one_of_many(abe_bad13.labeler_loss, "13 steps", title, self_label="20 steps")

    # plt.show()
    plt.savefig('/home/safoex/Documents/docs/writings/ambiguousobjectspaper/images/plots/aae_mustard_baseline_2.png')

# # models_dir = '/home/safoex/Downloads/cat_food/models_fixed/'
# # models_names = ['tonno_low', 'pollo', 'polpa']
# wdir_root = '/home/safoex/Documents/data/aae'
# models_dir = wdir_root + '/models/scans'
# # models_names = ['fragola', 'pistacchi', 'tiramisu']
# models_names = ['humana1', 'humana2']
# # models_names = ['melpollo', 'meltacchin']
# models = {mname: {'model_path': models_dir + '/' + mname + '.obj'} for mname in models_names}
# # workdir = wdir_root + '/test_labeler'
# model_name = 'babymilk'
# workdir = wdir_root + '/' + model_name + '2'
# if not os.path.exists(workdir):
#     os.mkdir(workdir)
# grider = Grid(300, 1)
#
# ae = AAE(128, 256, (128, 256, 256, 512))
# # ae_path = wdir_root + '/multi_boxes256.pth'
# ae_path = workdir + '/multi256.pth'
# ae.load_state_dict(torch.load(ae_path))
#
# ae.cuda()
#
# cl = torchvision.models.resnet18(pretrained=True).cuda()
# lat_size = cl.fc.in_features
# cl = torch.nn.Sequential(*list(cl.children())[:-1], torch.nn.Flatten())
#
# labeler = AmbigousObjectsLabeler(models, grider_label=grider, grider_codebook=Grid(1000, 10), ae=ae,
#                                  extra_encoder=cl, extra_latent_size=lat_size, extra_encoder_weight=0, ae_weight=1)
# # labeler.load_codebooks(workdir)
# labeler.load(workdir)
# # labeler.recalculate_extra_cl_sims()
# labeler.recalculate_fin_labels()
# # labeler.save(workdir)
# from matplotlib import pyplot as plt
#
# N = 300
# i_from = 0
# i_to = 1
# workdir_tops = workdir + '/tops_%d_%d/' % (i_from, i_to)
#
# obj1, obj2 = [], []
#
# from PIL import Image
# for i in range(N):
#     obj1.append(np.array(Image.open(workdir_tops + "top%d.png" % i)))
#     obj2.append(np.array(Image.open(workdir_tops + "top%d_f.png" % i)))
#
# loss = [np.linalg.norm(i1 - i2) / np.sum((i1 - i2) > 0) for i1, i2 in zip(obj1, obj2)]
#
# loss = np.load("/home/safoex/Downloads/babymilk.npy")
#
# labeler_res = np.sort(labeler._smoothen[:, i_from, i_to].cpu().numpy())
# scale_labeler = np.max(labeler_res) - np.min(labeler_res)
# scale_loss = np.max(loss) - np.min(loss)
#
# zero_labeler = np.min(labeler_res)
# zero_loss = np.min(loss)
# # plt.title("%s from obj %d to obj %d" % (model_name, i_from, i_to))
# # plt.plot( (loss - zero_loss) / scale_loss * scale_labeler + zero_labeler, label="scaled P2P loss")
# # plt.plot(labeler_res, label="AAE similarity")
# # plt.legend()
# # plt.show()
#
# plt.title("%s from obj %d to obj %d" % (model_name, i_from, i_to))
# plt.plot((loss - zero_loss) / scale_loss * scale_labeler + zero_labeler, label="scaled SIFT similarity")
# plt.plot(labeler_res, label="AAE similarity")
# plt.legend()
# plt.show()
#
# # for i_from, i_to in itertools.product(range(len(models)), range(len(models))):
# #     if i_from != i_to:
# #         top_n = 300
# #         wdir_save = workdir + '/' + 'tops_%d_%d' % (i_from, i_to)
# #         labeler.knn_median = 10
# #
# #         print_out_sim_views(labeler, models_names, models, i_from, i_to, top_n, wdir_save)
#
