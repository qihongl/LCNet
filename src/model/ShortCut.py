"""
a shortcut for predicting the context
"""
import torch
import torch.nn as nn
import numpy as np
from utils import to_sqpth, to_sqnp
from sklearn.model_selection import train_test_split


class ShortCut(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=128, lr=1e-3,
        buffer_size=512, allow_redundancy=True, precision=None):
        super().__init__()
        self.net = nn.Sequential(
                   nn.Linear(input_dim, hidden_dim),
                   nn.ReLU(),
                   nn.Linear(hidden_dim, output_dim),
                   nn.Sigmoid()
        )
        self.reset_weights()
        self.reset_buffer()
        self.buffer_size = buffer_size
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.prev_context = None
        self.allow_redundancy = allow_redundancy
        self.precision = precision

    def forward(self, X):
        return self.net(X)

    def add_data(self, x, y):
        # reformat to np
        if torch.is_tensor(x):
            x = to_sqnp(x)
        if torch.is_tensor(y):
            y = to_sqnp(y)

        if self.precision is not None:
            x = np.round(x, self.precision)

        # add data
        if self.allow_redundancy:
            self.X.append(x)
            self.Y.append(y)
        else:
            if self.in_buffer(x):
                first_match_id = np.where(self.comparison(x, self.X))[0][0]
                self.X.pop(first_match_id)
                self.Y.pop(first_match_id)
            self.X.append(x)
            self.Y.append(y)

        # check against buffer size
        if len(self.Y) > self.buffer_size:
            self.X.pop(0)
            self.Y.pop(0)

    def reset_weights(self):
        self.net.apply(init_weights)

    def reset_buffer(self):
        self.X = []
        self.Y = []

    def same(self, x1, x2):
        if self.precision is None:
            return np.isclose(x1, x2)
        return np.round(x1, self.precision) == np.round(x2, self.precision)

    def comparison(self, x_new, x_list):
        return [self.same(x_new, x_) for x_ in x_list]

    def in_buffer(self, x_new):
        return np.any(self.comparison(x_new, self.X))

    def replay_memory(self, n_epochs=1, test_ratio=.3, patience=10, re_train=True, n_examples_needed=10, verbose=False):
        if re_train:
            self.reset_weights()
        if len(self.Y) < n_examples_needed:
            return None
        # train test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=test_ratio)
        # data type
        # X_train = to_sqpth(torch.stack(X_train))
        X_train = to_sqpth(X_train)
        # X_test = to_sqpth(torch.stack(X_test))
        X_test = to_sqpth(X_test)
        Y_train = to_sqpth(np.array(Y_train))
        Y_test = to_sqpth(np.array(Y_test))

        # loop over epoch
        log_acc = np.zeros(n_epochs)
        for j in range(n_epochs):
            perm_ids = np.random.permutation(len(Y_train))
            loss = 0
            for i in perm_ids:
                # Yhat_i = self.net(X_train[i])
                loss += self.criterion(self.net(X_train[i]), Y_train[i])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            log_acc[j] = self.evaluate(X_test, Y_test)

            # if (j > patience and log_acc[j] < np.mean(log_acc[j-patience:j-1])) or log_acc[j] > .99:
            #     print(f'early stop at epoch {j}, acc = %.2f' % (log_acc[j]))
            #     break

        if verbose:
            print('Epoch: %.2d - Loss: %.2f - test acc: %.2f' % (j, loss, log_acc[j]))
        return log_acc[j]


    # def replay_memory(self, n_epochs=1, test_ratio=.3, patience=10, verbose=False):
    #     X_train = to_sqpth(self.X)
    #     Y_train = to_sqpth(np.array(self.Y))
    #     self.train_model(n_epochs, X_train, Y_train, verbose)
    #
    #
    # def train_model(self, n_epochs, X_train, Y_train, verbose=False):
    #     # loop over epoch
    #     log_acc = np.zeros(n_epochs)
    #     for j in range(n_epochs):
    #         perm_ids = np.random.permutation(len(Y_train))
    #         loss = 0
    #         for i in perm_ids:
    #             loss += self.criterion(self.net(X_train[i]), Y_train[i])
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         log_acc[j] = self.evaluate(X_train, Y_train)
    #
    #     if verbose:
    #         print('Epoch: %.2d - Loss: %.2f - train acc: %.2f' % (j, loss, log_acc[j]))



    @torch.no_grad()
    def infer_context(self, x):
        self.prev_context = self.net(x)
        # self.prev_context = context
        return self.prev_context

    def predict(self, X_test, use_array=False):
        X_test = to_sqpth(X_test)
        if not use_array:
            return np.array([to_sqnp(self.infer_context(X_test[i])) for i in range(len(X_test))])

        n_trials, n_time_steps = np.shape(X_test)[:2]
        Yhat = np.zeros(np.shape(X_test)[:2])
        for i in range(n_trials):
            for t in range(n_time_steps):
                ctx_it = self.infer_context(X_test[i, t])
                if ctx_it is not None:
                    Yhat[i,t] = to_sqnp(ctx_it)
        return Yhat


    def evaluate(self, X_test, Y_test):
        # X_test = to_sqpth(X_test)
        if torch.is_tensor(Y_test):
            Y_test = to_sqnp(Y_test)
        # Yhats = np.array([to_sqnp(self.infer_context(X_test[i])) for i in range(len(Y_test))])
        Yhats = self.predict(X_test)
        Yhats = np.round(Yhats)
        # print(Yhats)
        # print(Y_test)
        acc = np.mean(Yhats == Y_test)
        return acc



def init_weights(m, gain=1.0):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight, gain=gain)
        # torch.nn.init.xavier_uniform(m.weight, gain=gain)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import itertools
    sns.set(style='white', palette='colorblind', context='poster')
    # torch.manual_seed(0)
    # np.random.seed(0)

    data, labels = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)

    print(data.shape, labels.shape)

    # perm_ids = np.random.permutation(len(labels))
    # labels = labels[perm_ids]

    plt.scatter(data[:,0], data[:,1], c=labels)


    short_cut = ShortCut(input_dim=2, output_dim=1)

    for i in range(100):
        short_cut.add_data(data[i], labels[i])


    short_cut.replay_memory(n_epochs=99, verbose=True)

    # short_cut.reset_weights()
    acc = short_cut.evaluate(data, labels)
    print(acc)
