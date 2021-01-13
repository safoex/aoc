from matplotlib import pyplot as plt
import numpy as np
import pickle

workdir = '/home/safoex/Documents/data/aae/release2/release2'
classes = {'babyfood': ['meltacchin', 'melpollo'],
               'babymilk': ['humana1', 'humana2']}
def global_class_of(local_class, classes):
    for gcl, gcl_list in classes.items():
        if local_class in gcl_list:
            return gcl
    return None

extra_folder = 'local_%.2f' % 0.4

# with open(workdir + '/results_0.6.pickle', 'rb') as f:
#     results = pickle.load(f)

with open(workdir + '/'+  extra_folder + '/' + 'results.pickle', 'rb') as f:
    results = pickle.load(f)



av_results = []

for lcl, cl_results in results.items():
    for result in cl_results:
        if result is not None:
            av_results.append(((result[2] + result[3])/2, lcl == result[4]))


proc_results = {}
proc_funcs = {
    'average': np.average,
    'max': np.max,
    'min': np.min
}

for pf_name, pf in proc_funcs.items():
    av_results = []

    for lcl, cl_results in results.items():
        for result in cl_results:
            if result is not None:
                av_results.append((pf([result[2], result[3]]), lcl == result[4]))
    proc_results[pf_name] = av_results

def plot_results(some_results, title, threshold):
    X = np.linspace(0, 1, 100)
    Y = np.zeros_like(X)
    sigma = 0.05
    for i, x0 in enumerate(X):
        any_y = False
        for x, r in some_results:
            if x <= x0:
                any_y = True

        if any_y:
            Y[i] =  np.sum([int(r)  for x, r in some_results if x <= x0]) / \
                    np.sum([1       for x, r in some_results if x <= x0])

    plt.ylabel('correct classification probability')
    plt.xlabel('view ambiguity rank')
    plt.title('%s of dataset\ntaking %s of ambiguity ranks (assuming two different models)' % (threshold, title))
    plt.plot(X, Y, label='success probability for rank less than (x)')
    plt.plot(X, 0.5 * np.ones_like(X), 'r', label='0.5 prob baseline')
    plt.legend()
    plt.show()


for proc_name in proc_funcs:
    plot_results(proc_results[proc_name], proc_name, "3/5")