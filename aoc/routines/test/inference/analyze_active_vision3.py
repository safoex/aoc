from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

workdir = '/home/safoex/Documents/data/aae/release2/release2'
classes = {'babyfood': ['meltacchin', 'melpollo'],
           'babymilk': ['humana1', 'humana2']}
#
# classes = {'redboxes': ['tiramisu', 'pistacchi'],
#            'yellowboxes': ['cioccolato', 'vaniglia']}
#
# classes = {'yellowboxes': ['cioccolato', 'vaniglia']}
# classes = {'redboxes': ['tiramisu', 'pistacchi']}

# classes = {'babymilk': ['humana1', 'humana2']}

all_subclasses = sum([subcl for _, subcl in classes.items()], [])

def global_class_of(local_class, classes):
    for gcl, gcl_list in classes.items():
        if local_class in gcl_list:
            return gcl
    return None


pattern = "/home/safoex/Documents/data/finals/%s/%d/rad_0.30"

target_thresholds = np.linspace(0.05, 1, 20, endpoint=True)
jumps_limit = 4
max_jumps_positive = [np.zeros_like(target_thresholds) for _ in range(jumps_limit)]
max_jumps_total = [np.zeros_like(target_thresholds) for _ in range(jumps_limit)]

total_success = 0
total_success2 = 0
total_success1 = 0
total_successes = [0,0,0,0,0]
total = 0

all_threshs = []


for threshold in target_thresholds:
    objs = {}
    for n in range(4):
        for assumed_class in classes:
            for testclass in classes[assumed_class]:
                if testclass not in objs:
                    objs[testclass] = {
                        'positives' : [0 for _ in range(jumps_limit)],
                        'total': [0 for _ in range(jumps_limit)]
                    }
                # if assumed_class != 'babyfood':
                #     continue
                # if n in [ 3]:
                #     continue
                panda_dir = pattern % (testclass, n)
                with open(panda_dir + '/results.pickle', 'rb') as f:
                    results = pickle.load(f)

                for i, exp in enumerate(results):
                    # if testclass == 'melpollo':
                        # for rr in exp:
                        #     if rr[0] is not None:
                        #         print((rr[0][2] + rr[0][3])/2)
                        #     print(rr)
                        # print()
                    previous = i
                    target_achieved = False
                    good = False

                    if exp[0][0] is None:
                        break

                    for j in range(jumps_limit):
                        result = exp[j][0]

                        if result is None:
                            break

                        total_successes[j] += (exp[j][0][4] == testclass)
                        ambiguity = (result[2] + result[3]) / 2
                        est_class = result[4]

                        if ambiguity < threshold or previous == exp[j][1]:
                            if testclass == est_class:
                                if not target_achieved:
                                    good = True
                                for k in range(j, jumps_limit):
                                    objs[testclass]['positives'][k] += 1
                                    max_jumps_positive[k] += 1
                            target_achieved = True
                            for k in range(j, jumps_limit):
                                objs[testclass]['total'][k] += 1
                                max_jumps_total[k] += 1
                            break

                        previous = exp[j][1]
    all_threshs.append(objs)

plots = {}
convs = {}


for objs in all_threshs:
    for assumed_class in classes:
        objs[assumed_class] = {
                        'positives' : [0 for _ in range(jumps_limit)],
                        'total': [0 for _ in range(jumps_limit)]
                    }
        for lclass in classes[assumed_class]:
            for m in objs[lclass]:
                for j in range(len(objs[lclass][m])):
                    objs[assumed_class][m][j] += objs[lclass][m][j]

    for k, v in objs.items():
        if k not in plots:
            plots[k] = []
            convs[k] = []
        plots[k].append([pos/v['total'][-1] for pos, tot in zip(v['positives'], v['total'])])
        convs[k].append([1 - tot / v['total'][-1] for pos, tot in zip(v['positives'], v['total'])])
        # print(k, v)
    # print()

plt.figure(figsize=(4,4))
for gcl in list(classes.keys())[:1]:
    jumps = [[],[],[],[]]
    negs  = [[],[],[],[]]
    rbase = [[],[],[],[]]
    for p in plots[gcl]:
        for j, r in enumerate(p):
            jumps[j].append(r)
    for n in convs[gcl]:
        for j, r in enumerate(n):
            negs[j].append(r)
        rbase[0] = jumps[0]
    for j in range(1, jumps_limit):
        rbase[j] = [0 for _ in jumps[j]]
        print(jumps[j])
        neg = negs[0][0]
        for t in range(len(target_thresholds)):
            print(rbase[j][t])
            print(rbase[j - 1][t])
            print(jumps[j][t])
            rbase[j][t] = rbase[j - 1][t] + neg * jumps[0][t]
            neg -= neg * jumps[0][t]
    colors = ['tab:blue','tab:orange','tab:green','tab:red']
    for j, (jump, rbase_, color) in enumerate(zip(jumps, rbase, colors)):
        plt.plot(target_thresholds, jump,color=color, label=str(j) + ' extra moves', linewidth=2)
        if j >0:
            plt.plot(target_thresholds, rbase_,'.', color=color, linewidth=1)
plt.xlabel('threshold $rank(view) < x$')
plt.ylabel('correct classification performance')
titles = {'babyfood': 'boxes', 'babymilk': 'bottles'}
plt.title(titles[gcl])
# plt.legend()
plt.savefig('/home/safoex/Documents/docs/writings/ambiguousobjectspaper/images/plots/' + 'av_%s.png' % titles[gcl])
plt.show()
