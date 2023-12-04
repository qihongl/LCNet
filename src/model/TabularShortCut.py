"""
a TabularShortCut for predicting the context, use a lookup table
"""
import torch
# import torch.nn as nn
import numpy as np
from collections import Counter
from utils import to_sqpth, to_sqnp
# from sklearn.model_selection import train_test_split


class TabularShortCut():

    def __init__(self, input_dim, output_dim, precision=4, buffer_size=512, allow_redundancy=True):
        self.prev_context = 0
        self.X = []
        self.Y = []
        self.net = {}
        self.buffer_size = buffer_size
        self.precision = precision
        self.allow_redundancy = allow_redundancy
        self.data_count = 0

    def reset_buffer(self):
        self.X = []
        self.Y = []

    def round(self, x):
        return np.round(x, decimals=self.precision)

    def _add_data(self, x, y):
        self.X.append(x)
        self.Y.append(y)
        self.data_count +=1

    def _update_data(self, x, y):
        first_match_id = np.where(self.comparison(x, self.X))[0][0]
        self.X.pop(first_match_id)
        self.Y.pop(first_match_id)
        self._add_data(x, y)


    def add_data(self, x, y, update=False):
        # reformat to np
        if torch.is_tensor(x):
            x = to_sqnp(x)
        if torch.is_tensor(y):
            y = to_sqnp(y)
        if self.precision is not None:
            x = np.round(x, self.precision)
        # add data
        if self.allow_redundancy:
            self._add_data(x, y)
        else:
            # remove the old example
            if not self.in_buffer(x):
                self._add_data(x, y)
            else:
                if update:
                    # self._update_data(x, y)
                    self._add_data(x, y)
                # first_match_id = np.where(self.comparison(x, self.X))[0][0]
                # self.X.pop(first_match_id)
                # self.Y.pop(first_match_id)
                # self._add_data(x, y)
        # check against buffer size
        if len(self.Y) > self.buffer_size:
            self.X.pop(0)
            self.Y.pop(0)


    def same(self, x1, x2):
        if self.precision is None:
            return np.isclose(x1, x2)
        return np.round(x1, self.precision) == np.round(x2, self.precision)

    def comparison(self, x_new, x_list):
        return [self.same(x_new, x_) for x_ in x_list]

    def in_buffer(self, x_new):
        return np.any(self.comparison(x_new, self.X))


    def replay_memory(self, n_epochs=1, verbose=False):
        X = np.array(self.X)
        Y = np.array(self.Y)
        counter = {}
        # train test split
        unique_X = np.unique(X,axis=0)
        for x in unique_X:
            counter[str(x)] = []

        for x, y in zip(X, Y):
            counter[str(x)].append(y)

        for x in unique_X:
            x_counter = Counter(counter[str(x)])
            self.net[str(x)] = x_counter.most_common(1)[0][0]

        if verbose:
            print(f'number of unique obs: {len(counter.keys())}')
            print(f'labels for each obs: {counter.values()}')
            print(f'most common label for each obs: {self.net.values()}')
        # print(self.net.keys())
        # print(self.net.values())
        # return self.evaluate(X, Y)

    def infer_context(self, x):
        if torch.is_tensor(x):
            x = to_sqnp(x)
        x = self.round(x)
        if str(x) in self.net.keys():
            self.prev_context = self.net[str(np.array(x))]
        return self.prev_context

    def predict(self, X_test, use_array=True):
        if torch.is_tensor(X_test):
            X_test = to_sqnp(X_test)
        if not use_array:
            return np.array([self.infer_context(X_test[i]) for i in range(len(X_test))])

        n_trials, n_time_steps = np.shape(X_test)[:2]
        Yhat = np.zeros(np.shape(X_test)[:2])
        for i in range(n_trials):
            for t in range(n_time_steps):
                ctx_it = self.infer_context(X_test[i, t])
                if ctx_it is not None:
                    Yhat[i,t] = ctx_it
        return Yhat

    def evaluate(self, X_test, Y_test):
        # X_test = to_sqpth(X_test)
        if torch.is_tensor(Y_test):
            Y_test = to_sqnp(Y_test)

        Yhats = [self.infer_context(x) for x in X_test]
        Yhats = np.array(Yhats)

        acc = np.mean(Yhats == Y_test)
        return acc


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import itertools
    sns.set(style='white', palette='colorblind', context='poster')
    # torch.manual_seed(0)
    # np.random.seed(0)

    # [x1, x2] = np.random.normal(size=(2,3))
    [x1, x2] = [
        [0,0,0],
        [1,1,1],
    ]
    data = [x1, x1, x2, x1, x2]
    labels = [1, 1, 1, 0, 1]

    short_cut = TabularShortCut(input_dim=3, output_dim=1)
    short_cut.replay_memory(verbose=True)

    for i in range(len(data)):
        short_cut.add_data(data[i], labels[i])

    print(short_cut.X)
    print(short_cut.Y)

    short_cut.replay_memory(verbose=True)

    short_cut.infer_context(x1)
    acc = short_cut.evaluate(data, labels)

    # print(acc)
