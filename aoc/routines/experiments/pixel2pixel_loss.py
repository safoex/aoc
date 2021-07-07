from aoc.routines.experiments.aae_labeled_baseline import *

if __name__ == "__main__":
    wdir_root = '/home/safoex/Documents/data/aae'
    models_dir = wdir_root + '/models/scans'
    # models_names = ['humana1', 'humana2']
    models_names = ['melpollo', 'meltacchin']
    workdir = wdir_root + '/babyfood2'
    ae_path = workdir + '/multi256.pth'
    N = 300
    abe = AAEBaselineExperiment(models_dir, models_names, workdir, ae_path, N)

    obj1, obj2 = abe.get_images()

    pixel2pixel_loss = [1 - np.linalg.norm(i1 - i2) / np.sum( (i1 - i2) > 0  ) for i1, i2 in zip(obj1, obj2)]

    plt.figure(figsize=(4,5))
    abe.plot_comparison(pixel2pixel_loss, "MSE", "MSE", thickness=3)

    save_example_diff = False

    if save_example_diff:
        Image.fromarray(obj1[0]).show()
        Image.fromarray(obj2[0]).show()

        Image.fromarray(np.abs(obj1[0] - obj2[0])).show()
        # Image.fromarray(np.abs(obj1[-2] - obj2[-2])).show()
    plt.savefig('/home/safoex/Documents/docs/writings/ambiguousobjectspaper/images/plots/mse.png')
#
# wdir_root = '/home/safoex/Documents/data/aae/release2/babyfood'
# workdir = wdir_root + '/tops_1_0/'
#
# obj1, obj2 = [], []
#
# N = 200
#
# for i in range(N):
#     obj1.append(np.array(Image.open(workdir + "top%d.png" % i)))
#     obj2.append(np.array(Image.open(workdir + "top%d_f.png" % i)))
#
# loss = [np.linalg.norm(i1 - i2) / np.sum( (i1 - i2) > 0  ) for i1, i2 in zip(obj1, obj2)]
#
# for l in loss:
#     print(l)
#
# from matplotlib import pyplot as plt
#
# # AmbigousObjectsLabeler()
#
# plt.plot(loss)
# plt.show()
# #
# # print()
# # print(np.linalg.norm(obj1[0] - obj2[-1]))
# #
