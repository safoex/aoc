from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

workdir = '/home/safoex/Documents/data/aae/release2/release2'
classes = {'babyfood': ['meltacchin', 'melpollo'],
           'babymilk': ['humana1', 'humana2']}

# classes = {'babymilk': ['humana1', 'humana2']}

all_subclasses = sum([subcl for _, subcl in classes.items()], [])

def global_class_of(local_class, classes):
    for gcl, gcl_list in classes.items():
        if local_class in gcl_list:
            return gcl
    return None


extra_folder = 'local_%.2f' % 0.4

# with open(workdir + '/results_0.6.pickle', 'rb') as f:
#     results = pickle.load(f)

with open(workdir + '/' + extra_folder + '/' + 'results.pickle', 'rb') as f:
    results = pickle.load(f)


#
# av_results = []
#
# for lcl, cl_results in results.items():
#     for result in cl_results:
#         if result is not None:
#             av_results.append(((result[2] + result[3])/2, lcl == result[4]))
#

def proc_results_f(results_array):
    proc_results = {}
    proc_funcs = {
        'average': np.average,
        'max': np.max,
        'min': np.min
    }

    for pf_name, pf in proc_funcs.items():
        av_results = []

        for lcl, cl_results in results_array.items():
            if lcl in all_subclasses:
                for result in cl_results:
                    if result is not None:
                        av_results.append((pf([result[2], result[3]]), lcl == result[4]))
        proc_results[pf_name] = av_results
    return proc_results


proc_results = proc_results_f(results)


def plot_results(some_results, title, threshold, labels=True, baseline=True, color=None, legend=True, starting_from=10, fix_y_dims=(0.3, 1)):
    X = np.linspace(0, 1, 100)
    Y = np.zeros_like(X)
    sigma = 0.05
    for i, x0 in enumerate(X):
        any_y = False
        for x, r in some_results:
            if x <= x0:
                any_y = True

        if any_y:
            Y[i] = np.sum([int(r) for x, r in some_results if x <= x0]) / \
                   np.sum([1 for x, r in some_results if x <= x0])

    plt.ylabel('correct classification probability')
    plt.xlabel('for view ambiguity ranks less than X')
    if fix_y_dims is not None:
        plt.ylim(fix_y_dims)
    plt.title('%s of dataset\ntaking %s of ambiguity ranks (assuming two different models)' % (threshold, title))
    lbl = 'success probability for rank less than (x)'
    if starting_from > 0:
        X = X[starting_from:]
        Y = Y[starting_from:]
    if isinstance(labels, str):
        lbl = labels
    if labels is None:
        lbl = None
    if color:
        plt.plot(X, Y, color, label=lbl)
    else:
        plt.plot(X, Y, label=lbl)
    if baseline:
        plt.plot(X, 0.5 * np.ones_like(X), 'black', label='0.5 prob baseline' if labels else None)
    if legend:
        plt.legend()
    return Y


lessthan = [0.25, 0.4, 0.5, 0.7]

def plot_results_from_array(results_array, func_name, title, threshold, labels=None, color=None, legend=True,
                            starting_from=10, half_line=False, fix_y_dims=(0.3, 1)):
    if labels is None:
        labels = [None] * len(results_array)
    min_r = 1
    max_r = 0
    max_lessthan = np.zeros_like(lessthan)
    for i, (results, lbl) in enumerate(zip(results_array, labels)):
        Y = plot_results(proc_results_f(results)[func_name], title, threshold, lbl, baseline=(i == 0), color=color,
                         legend=legend, starting_from=starting_from)
        for i in range(len(lessthan)):
            max_lessthan[i] = max((Y[int(lessthan[i] * 100) - starting_from], max_lessthan[i]))
        min_r = min((min(Y), min_r))
        max_r = max((max(Y), max_r))

    if half_line:
        X = np.linspace(0, 1, 100)
        max_half = max_lessthan[2]
        plt.plot(X[starting_from:50], (max_half * np.ones(100))[starting_from:50], 'r--')
        y_vertical = np.linspace(min_r, max_half, 100)
        plt.plot(X[50] * np.ones(100), y_vertical, 'r--')

    return max_lessthan


# for proc_name in proc_funcs:
#     plot_results(proc_results[proc_name], proc_name, "3/5")
#
#     plt.show()


# results_file = '/home/safoex/Documents/data/results.pickle'
results_file = '/home/safoex/Documents/data/results_24_02_2021.pickle'

with open(results_file, 'rb') as f:
    results_dict = pickle.load(f)

# plot_results(proc_results_f(results_dict[2][0.4])['average'], 0.5, 'aga')
# figfolder = '/home/safoex/Documents/data/aae/draw/24'
# figpath = figfolder + "/inference2_0%d.png"

figfolder = '/home/safoex/Documents/docs/writings/ambiguousobjectspaper/images/plots/24'
figpath = figfolder + '/inference2_0%d.pdf'
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# threshold = 0.3
# epochs_range = list(range(2, 9))
# plot_results_from_array([results_dict[i][threshold] for i in epochs_range], 'average', 'average', threshold,
#                         labels=["after epoch=%d" % i for i in epochs_range], half_line=True)  # , color='blue')
# plt.show()

lessthan_performance = {k:list() for k in lessthan}
last_threshold_ = 1
last_threshold = int(last_threshold_ * 20)
for t in range(3, last_threshold):
    threshold = t / 20.0
    epochs_range = list(range(1, 9))
    perf_lessthan = plot_results_from_array([results_dict[i][threshold] for i in epochs_range], 'average', 'average', threshold,
                            labels=["after epoch=%d" % i for i in epochs_range], legend=True, half_line=True)#, color='red')
    for l, p in zip(lessthan, perf_lessthan):
        lessthan_performance[l].append(p)
    plt.savefig(figpath % int(threshold * 100))
    plt.clf()

threshold_list = np.array(list(range(3,last_threshold))) / 20.0

for k, perf in lessthan_performance.items():
    plt.plot(threshold_list, perf, label='r < %.1f' % k)
plt.legend()
plt.xlabel('trained on $rank(R^A_i) \leq x$ part of the dataset')
plt.ylabel('classification success rate')
plt.title('Performance of best among all epochs classifiers \ntested on $rank(R^A_i) \leq r$ part of recorded data')

plt.savefig(figfolder + '/sumup.pdf')
plt.savefig(figfolder + '/sumup.png')
# plt.show()