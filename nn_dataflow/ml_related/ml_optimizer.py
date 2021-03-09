import heapq
import sys
import time
import numpy as np
import random

class SimulatedAnnealingOptimizer(object):
    """Parallel simulated annealing optimization algorithm"""
    def __init__(self, design_space, n_iter=500, temp=(1, 0), persistent=True,
                 parallel_size=512, early_stop=50, log_interval=20):
        self.design_space = design_space
        self.n_iter = n_iter
        self.temp = temp
        self.persistent = persistent
        self.parallel_size = min(parallel_size, len(self.design_space))
        self.early_stop = early_stop or 1e9
        self.log_interval = log_interval
        self.points = None

    def find_maximums(self, model, num, exclusive):
        tic = time.time()
        temp, n_iter, early_stop, log_interval = \
                self.temp, self.n_iter, self.early_stop, self.log_interval

        if self.persistent and self.points is not None:
            points = self.points
        else:
            points = np.array(sample_ints(0, len(self.design_space), self.parallel_size))

        features = [self.design_space.index2feature(point) for point in points]
        # print("before tuplize: ")
        # print(features)
        # features = tuple(features)
        # print("after tuplize: ")
        # print(features)
        features = np.array(features)
        scores = model.predict(features)

        # convert scores to reciprocal to build a minimal heap.
        scores = np.array([1.0 / (item + 1e-6) for item in scores])

        # build heap and insert initial points
        heap_items = [(float('-inf'), - 1 - i) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = set(exclusive)
        in_heap.update([x[1] for x in heap_items])

        for s, p in zip(scores, points):
            if s > heap_items[0][0] and p not in in_heap:
                pop = heapq.heapreplace(heap_items, (s, p))
                in_heap.remove(pop[1])
                in_heap.add(p)

        k = 0
        k_last_modify = 0

        if isinstance(temp, (tuple, list, np.ndarray)):
            t = temp[0]
            cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)
        else:
            t = temp
            cool = 0

        while k < n_iter and k < k_last_modify + early_stop:
            new_points = np.empty_like(points)
            for i, p in enumerate(points):
                new_points[i] = random_walk(p, len(self.design_space))

            new_features = tuple(tuple(self.design_space.index2feature(point)) \
                            for point in points)
            new_scores = model.predict(new_features)

            ac_prob = np.exp(np.minimum((new_scores - scores) / (t + 1e-5), 1))
            ac_index = np.random.random(len(ac_prob)) < ac_prob
            ac_index = np.where(ac_index.astype(int)==1)

            points[ac_index] = new_points[ac_index]
            scores[ac_index] = new_scores[ac_index]

            for s, p in zip(new_scores, new_points):
                if s > heap_items[0][0] and p not in in_heap:
                    pop = heapq.heapreplace(heap_items, (s, p))
                    in_heap.remove(pop[1])
                    in_heap.add(p)
                    k_last_modify = k

            k += 1
            t -= cool

            if log_interval and k % log_interval == 0:
                t_str = "%.2f" % t
                # # print("SA iter: {} last_update: {} max-0: {} max-1: {} temp: {} elapsed: {}".format(
                #       k, k_last_modify, heap_items[0][0],
                #       np.max([v for v, _ in heap_items]), t_str,
                #       time.time() - tic))

        heap_items.sort(key=lambda item: -item[0])
        heap_items = [x for x in heap_items if x[0] >= 0]
        # sys.stderr.write("SA iter: {} last_update: {} elapsed: {}\n".format(k, k_last_modify, time.time() - tic))
        # print("SA Maximums: {}".format(heap_items))

        if self.persistent:
            self.points = points

        return [x[1] for x in heap_items]


def sample_ints(low, high, m):
    """
    Sample m different integer numbers from [low, high) without replacement
    This function is an alternative of `np.random.choice` when (high - low) > 2 ^ 32, in
    which case numpy does not work.
    Parameters
    ----------
    low: int
        low point of sample range
    high: int
        high point of sample range
    m: int
        The number of sampled int
    Returns
    -------
    ints: an array of size m
    """
    vis = set()
    assert m <= high - low
    while len(vis) < m:
        new = random.randrange(low, high)
        while new in vis:
            new = random.randrange(low, high)
        vis.add(new)

    return list(vis)


def random_walk(p, dim_len):
    """random walk as local transition
    Parameters
    ----------
    p: int
    Returns
    -------
    new_p: int
        new neighborhood index
    """
    # mutate
    new_p = p
    while new_p == p:
        new_p = min(max(int(random.gauss(p, 1)), 0), dim_len-1)

    # transform to index form
    return new_p
