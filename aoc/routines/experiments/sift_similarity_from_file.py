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

    sift_loss = np.load("/home/safoex/Downloads/babyfood.npy")

    #pixel2pixel_loss = [1 - np.linalg.norm(i1 - i2) / np.sum( (i1 - i2) > 0  ) for i1, i2 in zip(obj1, obj2)]

    plt.figure(figsize=(4,5))
    abe.plot_comparison(sift_loss, "SIFT", "SIFT", thickness=3)
    # plt.show()
    plt.savefig('/home/safoex/Documents/docs/writings/ambiguousobjectspaper/images/plots/sift.png')
