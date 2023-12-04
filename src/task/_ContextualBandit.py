import numpy as np
# import itertools
from copy import deepcopy
from utils import to_pth, to_np
from itertools import groupby, chain

context_types = ['null', 'truth']
dtypes = ['int', 'float']


class ContextualBandit():
    """Short summary.

    Parameters
    ----------
    n_machine : type
        Description of parameter `n_machine`.
    p : type
        Description of parameter `p`.
    n_arm : type
        Description of parameter `n_arm`.
    T_min : type
        Description of parameter `T_min`.
    T_max : type
        Description of parameter `T_max`.
    percept_dim : type
        Description of parameter `percept_dim`.
    percept_type : str
        'null' - a fixed random vector, which is uninformative
        'truth' - always provide the ground truth percept vector

    Attributes
    ----------
    percept : type
        Description of attribute `percept`.
    n_machine
    p
    n_arm
    T_min
    T_max
    percept_dim
    percept_type

    """

    def __init__(
            self, n_machine, p,
            n_arm=2,
            T_min=3,
            T_max=7,
            percept_dim=16,
            percept_type='null',
            noise=False,
            noise_level=.1,
            dtype='float',

    ):
        assert np.shape(p) == (n_machine, n_arm), 'inconsistent input'
        assert np.allclose(np.sum(p, axis=1), np.ones(n_machine))
        assert T_max >= T_min > 0
        assert percept_dim > 0, 'lol'
        assert percept_type in context_types
        # params
        self.n_machine = n_machine
        self.p = p
        self.n_arm = n_arm
        self.T_min = T_min
        self.T_max = T_max
        self.percept_dim = percept_dim
        self.percept_type = percept_type
        self.dtype = dtype
        self.percept = [
            np.random.normal(size=self.percept_dim) for _ in range(n_machine)
        ]
        self.noise_level = noise_level
        self.noise = noise
        self.shuffle_machine_order()
        assert dtype in dtypes, 'dtype must be int or float'
        if dtype == 'int':
            for i, prcpt_i in enumerate(self.percept):
                prcpt_i[prcpt_i > 0] = 1
                prcpt_i[prcpt_i <= 0] = 0

    def shuffle_machine_order(self):
        perm = np.random.permutation(range(self.n_machine))
        self.p = [self.p[k] for k in perm]
        self.percept = [self.percept[k] for k in perm]

    def sample(self, fix_seq_len=-1, format_data=True):
        raw_targets, machine_ids, event_bounds = [], [], []
        for i in range(self.n_machine):
            # construct x and y for an episode
            raw_targets_i, machine_ids_i, T_i = self._sample_machine_i(
                i, fix_seq_len)
            raw_targets.extend(raw_targets_i)
            machine_ids.extend(machine_ids_i)
            event_bounds.extend([T_i])
        machine_order = [int(x[0]) for x in groupby(machine_ids)]
        percept = self._make_percepts(machine_order, event_bounds)
        # reformat data
        if format_data:
            X, Y = self._format_xy(percept, raw_targets)
            machine_ids = np.array(machine_ids)
            return X, Y, machine_ids, event_bounds
        return percept, raw_targets, machine_ids, event_bounds

    def _sample_machine_i(self, context_id, fix_seq_len=-1):
        if fix_seq_len == -1:
            T_i = int(np.random.uniform(low=self.T_min, high=self.T_max))
        else:
            assert fix_seq_len % self.n_machine == 0
            T_i = fix_seq_len // self.n_machine
        # sample the correct arms from the i-th machine
        targets_i = np.random.choice(
            np.arange(self.n_arm), size=T_i, p=self.p[context_id]
        )
        machine_ids_i = np.ones(T_i) * context_id
        return targets_i, machine_ids_i, T_i

    def _make_percepts(self, machine_order, Ts):
        # compute the sequence lenght for the entire episode
        T_total = np.sum(Ts)
        # form the percept inputs
        if self.percept_type == 'null':
            percept = [
                self.add_noise(self.percept[0])
                for _ in range(T_total)
            ]
            [self.percept[0]] * T_total
        elif self.percept_type == 'truth':
            percept = list(chain.from_iterable(
                [
                    [
                        self.add_noise(self.percept[machine_id])
                        for _ in range(Ts[machine_id])
                    ]
                    for machine_id in machine_order
                ]
            ))
        else:
            raise ValueError()
        return percept

    def add_noise(self, input_percept):
        if not self.noise:
            return input_percept
        if self.dtype == 'int':
            noisy_percept = deepcopy(input_percept)
            n_bit_flip = int(np.round(self.noise_level * self.percept_dim))
            bit_flip = np.random.choice(
                range(self.percept_dim), n_bit_flip, replace=False
            )
            for bf_i in bit_flip:
                noisy_percept[bf_i] = self.flip(noisy_percept[bf_i])
            return noisy_percept
        elif self.dtype == 'float':
            raise ValueError('Q has not implement it ')
        else:
            raise ValueError('dtype must be int or float')

    def flip(self, x):
        if x == 0:
            return 1
        elif x == 1:
            return 0
        else:
            raise ValueError('x to be fliped, must be 0 or 1')

    def _format_xy(self, percept, raw_targets):
        # reformat the data to NN-readable
        X = to_pth(np.array(percept))
        Y = to_pth(np.hstack(raw_targets))
        return X, Y

    def __repr__(self):
        _repr = f'n_machine = {self.n_machine}\n'
        _repr += f'n_arm = {self.n_arm}\n'
        _repr += f'n_time_step = ({self.T_min, self.T_max})\n'
        _repr += 'p = ' + str(self.p)
        return _repr


'''how to use'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', palette='colorblind', context='talk')
    np.random.seed(1)

    n_machine = 2
    p = np.array([[.05, .95], [.95, .05]])
    # p = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    n_arm = 2
    T_min = 1
    T_max = 5
    percept_dim = 16
    percept_type = 'truth'
    # initialize the task
    bandit = ContextualBandit(
        n_machine, p, n_arm,
        T_min=T_min, T_max=T_max,
        percept_dim=percept_dim, percept_type=percept_type, dtype='int',
        noise=True,
        noise_level=0,
    )

    # # sample a trial
    # print('Raw data:')
    # X, Y, machine_ids = bandit.sample(format_data=False)
    # print(X)
    # print(Y)
    # print(machine_ids)

    # np.sum(machine_ids == i) for i in range()

    print('Formatted:')
    X, Y, machine_ids, event_bounds = bandit.sample(
        fix_seq_len=12, format_data=True)
    print(X)
    print(Y)
    # print(len(Y))
    # print(machine_ids)
    # print(event_bounds)
    sns.heatmap(X, linewidth=1)
    # # bandit.percept[0]
