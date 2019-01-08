import numpy as np


class StationaryKbandit:
    def __init__(self, k=10):
        self._k = k
        self._action_value = np.random.normal(loc=0, scale=1, size=self._k)

    def step(self, num):
        mean_reward = self._action_value[num]
        reward = np.random.normal(loc=mean_reward, scale=1)
        return reward

    def ideal_arm(self):
        return np.max(self._action_value), np.argmax(self._action_value)


class NonStationaryKbandit:
    def __init__(self, k=10):
        self._k = k
        init_action_val = np.random.normal(loc=0, scale=1)
        self._action_value = np.full(shape=self._k, fill_value=init_action_val, dtype=np.float64)
        self._random_walk = np.random.normal(loc=0, scale=0.01, size=self._k)

    def step(self, num):
        mean_reward = self._action_value[num]
        reward = np.random.normal(loc=mean_reward, scale=1)
        self._random_walk = np.random.normal(loc=0, scale=0.01, size=self._k)
        self._action_value = self._action_value + self._random_walk
        return reward

    def ideal_arm(self):
        return np.max(self._action_value), np.argmax(self._action_value)
