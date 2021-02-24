from mitmpose.model.classification.labeler_ambigous import AmbigousObjectsLabeler, AAE, Grid
import os
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

from mitmpose.model.classification.labeler_ambigous import AmbigousObjectsLabeler, AAE, Grid
import os
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt


class AAEBaselineExperiment:
    def __init__(self, models_dir, models_names, workdir, ae_path, n_grid=300):
        self.n = n_grid
        models = {mname: {'model_path': models_dir + '/' + mname + '.obj'} for mname in models_names}
        ae = AAE(128, 256, (128, 256, 256, 512))
        ae.load_state_dict(torch.load(ae_path))
        grider = Grid(n_grid, 1)

        labeler = AmbigousObjectsLabeler(models, grider_label=grider, grider_codebook=Grid(1000, 10), ae=ae,
                                         extra_encoder_weight=0, ae_weight=1)
        labeler.load(workdir)
        labeler.recalculate_fin_labels()

        N = n_grid
        i_from = 0
        i_to = 1
        workdir_tops = workdir + '/tops_%d_%d/' % (i_from, i_to)

        self.obj1, self.obj2 = [], []

        for i in range(N):
            self.obj1.append(np.array(Image.open(workdir_tops + "top%d.png" % i)))
            self.obj2.append(np.array(Image.open(workdir_tops + "top%d_f.png" % i)))

        self.labeler_loss = np.sort(labeler._smoothen[:, i_from, i_to].cpu().numpy())

        self.ae = ae
        self.models = models
        self.grider = grider
        self.labeler = labeler

        self.plotted_once = False

    def get_images(self):
        return self.obj1, self.obj2

    def get_indices(self, i_from, i_to):
        return torch.topk(-self.labeler._sorted[:, i_from, i_to], self.n)[1].cpu().numpy()

    def plot_solo(self, title_name):
        plt.title(title_name)
        plt.plot(self.labeler_loss, label="AAE similarity")
        plt.legend()
        plt.show()

    def plot_comparison(self, loss, similarity_name, title_name):
        scale_labeler = np.max(self.labeler_loss) - np.min(self.labeler_loss)
        scale_loss = np.max(loss) - np.min(loss)

        zero_labeler = np.min(self.labeler_loss)
        zero_loss = np.min(loss)

        plt.title(title_name)
        plt.plot((loss - zero_loss) / scale_loss * scale_labeler + zero_labeler,
                 label="scaled %s similarity" % similarity_name)
        plt.plot(self.labeler_loss, label="AAE similarity")
        plt.legend()
        plt.show()

    def plot_one_of_many(self, loss, similarity_name, title_name):
        scale_labeler = np.max(self.labeler_loss) - np.min(self.labeler_loss)
        scale_loss = np.max(loss) - np.min(loss)

        zero_labeler = np.min(self.labeler_loss)
        zero_loss = np.min(loss)

        plt.plot((loss - zero_loss) / scale_loss * scale_labeler + zero_labeler,
                 label="scaled %s similarity" % similarity_name)

        if not self.plotted_once:
            plt.title(title_name)
            plt.plot(self.labeler_loss, label="AAE similarity")
            self.plotted_once = True

        plt.legend()

    def resort_pairs(self, loss, new_workdir):
        t_loss = torch.from_numpy(loss)
        tops, idcs = torch.topk(t_loss, self.n)

        if not os.path.exists(new_workdir):
            os.mkdir(new_workdir)

        for i_new, i_old in enumerate(idcs):
            Image.fromarray(self.obj1[i_old]).save(new_workdir + "tops%d.png" % i_new)
            Image.fromarray(self.obj2[i_old]).save(new_workdir + "tops%d_f.png" % i_new)


if __name__ == "__main__":
    wdir_root = '/home/safoex/Documents/data/aae'
    models_dir = wdir_root + '/models/scans'
    # models_names = ['humana1', 'humana2']
    # models_names = ['melpollo', 'meltacchin']
    models_names = ['mustardblue', 'mustardgreen']
    # workdir = wdir_root + '/babymilk2'
    workdir = wdir_root + '/mustard'
    ae_path = workdir + '/multi256.pth'
    N = 100
    abe = AAEBaselineExperiment(models_dir, models_names, workdir, ae_path, N)

    abe.plot_solo("aae solo")

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
