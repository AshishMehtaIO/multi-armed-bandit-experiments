import numpy as np


class EGreedySampleAvgRewAgent(object):
    def __init__(self, k, epsilon, _):
        self._k = k
        self._Q = np.zeros(shape=self._k)
        self._N = np.zeros(shape=self._k)
        self._epsilon = epsilon

    def __str__(self):
        return 'EGreedySampleAvgRewAgent'

    def sample_action(self):
        # argmax Q with prob 1-e
        if np.random.uniform() < (1 - self._epsilon):
            return np.argmax(self._Q)
        # random action with prob e
        else:
            return np.random.randint(low=0, high=self._k)

    def update_value(self, reward, act):
        self._N[act] = self._N[act] + 1
        self._Q[act] = self._Q[act] + (1 / self._N[act]) * (reward - self._Q[act])


class ConstStepRewAgent(object):
    def __init__(self, k, epsilon, alpha):
        self._k = k
        self._Q = np.zeros(shape=self._k)
        self._N = np.zeros(shape=self._k)
        self._epsilon = epsilon
        self._alpha = alpha

    def __str__(self):
        return 'ConstStepRewAgent'

    def sample_action(self):
        # argmax Q with prob 1-e
        if np.random.uniform() < (1 - self._epsilon):
            return np.argmax(self._Q)
        # random action with prob e
        else:
            return np.random.randint(low=0, high=self._k)

    def update_value(self, reward, act):
        self._N[act] = self._N[act] + 1
        self._Q[act] = self._Q[act] + self._alpha * (reward - self._Q[act])


class OptimisticInitialValuesGreedyAgent(object):
    def __init__(self, k, epsilon, _):
        self._k = k
        self._Q = np.full(shape=self._k, fill_value=5.0)
        self._N = np.zeros(shape=self._k)
        self._epsilon = epsilon

    def __str__(self):
        return 'OptimisticInitialValuesGreedyAgent'

    def sample_action(self):
        # always greedy
        return np.argmax(self._Q)

    def update_value(self, reward, act):
        self._N[act] = self._N[act] + 1
        self._Q[act] = self._Q[act] + (1 / self._N[act]) * (reward - self._Q[act])


# TODO check Unbiased Constant-Step-Size Trick
class UnbiasedConstStepAgent(object):
    def __init__(self, k, epsilon, alpha):
        self._k = k
        self._Q = np.zeros(shape=self._k)
        self._N = np.zeros(shape=self._k)
        self._epsilon = epsilon
        self._alpha = alpha
        self._trace = 0

    def __str__(self):
        return 'UnbiasedConstStepAgent'

    def sample_action(self):
        # argmax Q with prob 1-e
        if np.random.uniform() < (1 - self._epsilon):
            return np.argmax(self._Q)
        # random action with prob e
        else:
            return np.random.randint(low=0, high=self._k)

    def update_value(self, reward, act):
        self._N[act] = self._N[act] + 1
        self._trace = self._trace + self._alpha * (1 - self._trace)
        self._Q[act] = self._Q[act] + (self._alpha/self._trace) * (reward - self._Q[act])

#
# class UpperConfidenceBoundAgent:


