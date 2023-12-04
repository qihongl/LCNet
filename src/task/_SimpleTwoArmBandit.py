import numpy as np
from utils import to_pth

N_ARM = 2


class SimpleTwoArmBandit():

    def __init__(self, p0, reward=1, penalty=0, n_tps=10):
        assert 0 <= p0 <= 1
        assert reward > 0 and penalty <= 0
        self.p0 = p0
        self.reward = reward
        self.penalty = penalty

    def get_reward(self, action_t):
        if action_t not in [0, 1, 2]:
            raise ValueError('invalid action')
        u = np.random.uniform(low=0, high=1)
        if action_t == 0:
            # reward the model for right answer with probability p
            return self.reward if u < self.p0 else self.penalty
        elif action_t == 1:
            # punish ...
            return self.reward if u < (1 - self.p0) else self.penalty
        else:
            # if the model says "don't know", then r = 0
            return 0


if __name__ == '__main__':
    np.random.seed(0)

    p0 = 1
    bandit = SimpleTwoArmBandit(p0, reward=1, penalty=-1)

    a_t = 0
    n_tps = 20
    r = np.empty(n_tps,)
    for t in range(n_tps):
        r[t] = bandit.get_reward(a_t)

    '''cumulative reward for optimal exploitation '''
    def run_exploitation_exp(p0):

        bandit = SimpleTwoArmBandit(p0, reward=1, penalty=-1.5)

        a_t = 0
        n_tps = 100
        r = np.empty(n_tps,)
        for t in range(n_tps):
            r[t] = bandit.get_reward(a_t)
        return np.mean(r)

    ps = np.arange(.5, 1.01, .1)
    cumr = [run_exploitation_exp(p0) for p0 in ps]

    print(cumr)
