import numpy as np
from numpy.random import default_rng

class Firefly:
    def __init__(self, pop_size=20, alpha=1.0, betamin=1.0, gamma=0.01, seed=None):
        self.pop_size = pop_size
        self.alpha = alpha
        self.betamin = betamin
        self.gamma = gamma
        self.rng = default_rng(seed)

    def run(self, function, dim, lowerbound, upperbound, max_evals, pop_size):
        fireflies = self.rng.uniform(lowerbound, upperbound, (pop_size, dim))
        intensity = np.apply_along_axis(function, 1, fireflies)
        # print(intensity)
        bestvals=[]
        best = np.min(intensity)
        # print(best)
        evaluations = self.pop_size
        new_alpha = self.alpha
        search_range = upperbound - lowerbound

        while evaluations <= max_evals:
            new_alpha *= 0.97
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if intensity[i] >= intensity[j]:
                        brightness=fireflies[i] - fireflies[j]
                        attractiveness = np.sum(np.square(brightness), axis=-1)
                        beta = self.betamin * np.exp(-self.gamma * attractiveness)
                        steps = new_alpha * (self.rng.random(dim) - 0.5) * search_range
                        # print(self.rng.random(dim))
                        fireflies[i] += beta * (brightness) + steps
                        fireflies[i] = np.clip(fireflies[i], lowerbound, upperbound)
                        intensity[i] = function(fireflies[i])
                        # print(fireflies[i],intensity[i])
                        evaluations += 1
                        best = min(intensity[i], best)
                        bestvals.append(best)
        return bestvals,best
