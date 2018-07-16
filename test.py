from orig.meteor.meteor import Meteor
from time import time

cands = ["An eagle perched among trees", "Eagles perched in trees", "Bird perched on tree"]
refs = ["A bald eagle sits on a perch",
        "An american bald eagle sitting on a branch in the zoo",
        "Bald eagle perched on piece of lumber",
        "A large bird standing on a tree branch"]

scorer = Meteor()

for cand in cands:

    x = {1: [cand]}
    y = {1: refs}

    start = time()
    scorer.compute_score(y, x)
    print(time() - start)


cands = ["A bird pulls food out of a bird feeder", "A bird is feeding at a bird feeder", "A colorful bird with his head inside of a bird feeder"]
refs = ["A bird is looking inside a birdhouse",
        "A bird is eating from a bird feeder",
        "A bird eating from a bird feeder",
        "A red-tailed bird eats from a bird feeder. "]

scorer = Meteor()

for cand in cands:

    x = {1: [cand]}
    y = {1: refs}

    start = time()
    scorer.compute_score(y, x)
    print(time() - start)

for i in range(10):
    x = {1: [cands[0]]}
    y = {1: refs}

    start = time()
    scorer.compute_score(y, x)
    print(time() - start)