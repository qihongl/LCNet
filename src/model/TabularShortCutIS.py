"""
a TabularShortCut for predicting the context, use a lookup table
input specific
"""
import torch
# import torch.nn as nn
import numpy as np
from collections import Counter
from utils import to_sqpth, to_sqnp

class TabularShortCutIS():

    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.reset()

    def reset(self):
        self.X = []
        self.Y = []
        self.buffer = {}
        self.prev_context = 0
        self.data_count = 0

    def _add_to_XY(self, x, y):
        self.X.append(x)
        self.Y.append(y)
        self.data_count += 1


    def _add_data(self, x, y):
        tuple_x = tuple(x)
        if tuple_x in self.buffer.keys():
            if not self.n_associates(x) == 1 and self.buffer[tuple_x][0] == y:
                self._add_to_XY(x, y)
        # if new input, allocate an empty list
        if tuple_x not in self.buffer.keys():
            self.buffer[tuple_x] = []
        # append the new full inference result
        self.buffer[tuple_x].append(y)
        # pop the earlier assoc if full
        if len(self.buffer[tuple_x]) > self.buffer_size:
            self.buffer[tuple_x].pop(0)


    # def _update_data(self, x, y):
    #     first_match_id = np.where(self.comparison(x, self.X))[0][0]
    #     self.X.pop(first_match_id)
    #     self.Y.pop(first_match_id)
    #     self._add_data(x, y)


    def add_data(self, x, y, update=False):
        # reformat to np
        if torch.is_tensor(x):
            x = to_sqnp(x)
        if torch.is_tensor(y):
            y = to_sqnp(y)
        # add data
        # if update:
        #     self._update_data(x, y)
        # else:
        #     self._add_data(x, y)
        if update:
            self._add_data(x, y)


    def is_novel(self, x):
        return tuple(x) not in self.buffer.keys()

    def n_associates(self, x):
        return len(np.unique(self.buffer[tuple(x)]))


    def infer_context(self, x):
        if torch.is_tensor(x):
            x = to_sqnp(x)
        # if novel input
        if self.is_novel(x):
            return self.prev_context, None
        # cache the prev context
        prev_context = self.prev_context
        # self.prev_context = None
        # if there is only one associated context
        if self.n_associates(x) == 1:
            self.prev_context = self.buffer[tuple(x)][0]
        # self.prev_context = self._infer_context(x)
        # return previous context and the shortcut context
        return prev_context, self.prev_context

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
    labels = [1, 0, 0, 0, 0]

    short_cut = TabularShortCutIS()
    # short_cut.replay_memory(verbose=True)

    print(f'x1 is novel: {short_cut.is_novel(x1)}')

    for i in range(len(data)):
        print(i)
        short_cut.add_data(data[i], labels[i])
        print(short_cut.buffer)

    print(short_cut.buffer.keys())
    print()
    print(f'x1 is novel: {short_cut.is_novel(x1)}')


    # print(short_cut.X)
    # print(short_cut.Y)


    # short_cut.replay_memory(verbose=True)

    # short_cut.infer_context(x1)
    # acc = short_cut.evaluate(data, labels)
    #
    # # print(acc)
