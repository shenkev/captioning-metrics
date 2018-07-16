from preprocess_cider import build_dict
from ruotian.ciderD.ciderD import CiderD
from orig.bleu.bleu import Bleu
from orig.rouge.rouge import Rouge
from orig.meteor.meteor import Meteor
from orig.spice.spice import Spice
from time import time
import numpy as np

cand = "An eagle perched among trees"
refs = ["A bald eagle sits on a perch",
        "An american bald eagle sitting on a branch in the zoo",
        "Bald eagle perched on piece of lumber",
        "A large bird standing on a tree branch"]

# build_dict(refs)

x = {1: [cand]}
y = {1: refs}


def profile_metric(scorer, name, y, x, trials=100):
    times = []

    for i in range(trials):
        start = time()
        scorer.compute_score(y, x)
        times.append(time()-start)

    times = np.asarray(times)

    print "==================== Metric {} ====================".format(name)
    print "Avg: {}ms \t Max: {}ms \t Min: {}ms \t Std: {}ms \t".format(1000.0*np.average(times), 1000.0*np.max(times), 1000.0*np.min(times), 1000.0*np.std(times))
    print times

scorers = [[Bleu(), "bleu"], [Rouge(), "rouge"], [Meteor(), "meteor"], [Spice(), "spice"]]

for scorer, name in scorers:
    profile_metric(scorer, name, y, x)


x = [{'image_id': 1, 'caption': [cand]}]
y = {1: refs}

ciderD_scorer = CiderD(df='./cider-idxs.p')
profile_metric(ciderD_scorer, "cider", y, x)