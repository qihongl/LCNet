''' using an rnn with context input solve the alternating sin-cos task
the ground truth context is provided (assuming CRP works perfectly)

next steps:
1. use CRP to monitor hidden states and PE to produce event boundaries
2. explore tasks with share structure
'''
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from utils import to_np, to_pth, to_sqnp
from model import CRPLSTM as Agent
from model import dLocalMAP as CRP
from task import Waves
from stats import compute_stats
from vis import plot_learning_curve, make_line_plot_wer

sns.set(style='white', palette='colorblind', context='talk')
cpal = sns.color_palette()
print(f'GPU avail: {torch.cuda.is_available()}')

# set seed
seed_val = 0
torch.manual_seed(seed_val)
np.random.seed(seed_val)

# env param
noise = 0.03
start = -4
stop = 4
step = 1 / 3
task = Waves(percept_dim=32, noise=noise, start=start, stop=stop, step=step)
x, y, y_label, percept = task.get_data(to_pth=True)
T = len(x) - 1

# model param
n_epochs = 1000
learning_rate = 3e-4
dim_input = 2
dim_hidden = 64
dim_context = dim_hidden
dim_output = 1

# CRP param
pe_threshold = .4
c = .01
lmbd = 0

# init model
crp = CRP(c=c, lmbd=lmbd, context_dim=dim_context,
          allow_redundant_instance=False)
agent = Agent(
    dim_input, dim_hidden, dim_output, dim_context, output_format='continuous',
    use_ctx=True
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

# train the model
# i,t=0,0

log_loss = np.zeros(n_epochs)
for i in range(n_epochs):
    loss = 0
    pe_it_next = 0
    h_t = torch.zeros(1, 1, dim_hidden)
    c_t = torch.zeros(1, 1, dim_hidden)
    x, y, y_label, percept = task.get_data(noise_on=True, to_pth=True)

    for t in np.arange(0, task.T - 1):

        # c_i = crp.stimulate(
        #     np.array(np.round(to_np(percept[t + 1])), dtype=np.int))
        c_i = crp.stimulate(
            np.array(to_sqnp(torch.round(h_t)), dtype=np.int)
        )
        context_t = to_pth(crp.context[c_i])

        # given y_t (t from 0 to T-1) predict y_hat_t+1 (from 1 to T)
        input_t = torch.tensor([y[t], pe_it_next])
        [yhat_it_next, _, _, h_t, c_t], cache = agent.forward(
            input_t.view(1, 1, -1), h_t, c_t, context_t
        )
        pe_it_next = criterion(yhat_it_next, y[t + 1])
        loss += pe_it_next
    # update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # record info
    log_loss[i] = to_np(loss)
    if i % 10 == 0:
        print('Epoch %3d | loss = %6.3f' % (i, log_loss[i]))

n_epochs_test = 50
Y_label = np.zeros((n_epochs_test, task.T))
Y = torch.zeros((n_epochs_test, task.T))
Yhat = torch.zeros((n_epochs_test, task.T - 1))
H = torch.zeros((n_epochs_test, task.T, dim_hidden))
PE = torch.zeros((n_epochs_test, task.T - 1))
C = np.zeros((n_epochs_test, task.T - 1))
for i in range(n_epochs_test):
    pe_it_next = 0
    h_t = torch.zeros(1, 1, dim_hidden)
    c_t = torch.zeros(1, 1, dim_hidden)
    x, Y[i, :], Y_label[i, :], percept = task.get_data(to_pth=True)

    for t in np.arange(0, task.T - 1):

        # C[i, t] = crp.stimulate(
        #     np.array(np.round(to_np(percept[t + 1])), dtype=np.int))
        C[i, t] = crp.stimulate(
            np.array(to_sqnp(torch.round(h_t)), dtype=np.int))
        context_t = to_pth(crp.context[int(C[i, t])])

        input_t = torch.tensor([Y[i, t], pe_it_next])
        [Yhat[i, t], _, _, h_t, c_t], cache = agent.forward(
            input_t.view(1, 1, -1), h_t, c_t, context_t
        )
        pe_it_next = criterion(Yhat[i, t], Y[i, t + 1])
        # log info
        PE[i, t] = pe_it_next
        H[i, t + 1] = h_t

# data to np, remove the 1st event
error = np.abs(to_np(PE[:, task.event_len:]))
Yhat = to_np(Yhat[:, task.event_len:])
H = to_np(H[:, task.event_len + 1:, :])
Y = to_np(Y[:, task.event_len + 1:])
x = to_np(x[task.event_len + 1:])
Y_label = Y_label[:, task.event_len + 1:]

# PCA the hidden states
n_components = 2
pca = PCA(n_components=n_components)
H2d_rs = pca.fit_transform(np.reshape(H, (-1, dim_hidden)))
H2d = np.reshape(H2d_rs, (np.shape(H)[0], np.shape(H)[1], n_components))


# plot learning curve
f, ax = plot_learning_curve(
    np.log(log_loss), ylabel='log(MSE)', figsize=(6, 4), window_size=5,
)

# plot single trial performance
n_trial_plot = 10
for i in range(n_trial_plot):
    f, axes = plt.subplots(2, 1, figsize=(5, 6))
    # plot Y
    axes[0].plot(x[Y_label[i] == 0], Y[i][Y_label[i] == 0], '.')
    axes[0].plot(x[Y_label[i] == 1], Y[i][Y_label[i] == 1], '.')
    # plot Yhat
    axes[0].plot(x, Yhat[i], label='y_hat', color='k')
    axes[1].stem(error[i])
    axes[0].set_ylabel('y')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('|PE|')
    axes[0].legend()
    # axes[1].set_ylim([0, np.max(error) * 1.2])
    axes[1].set_ylim([0, 1.1])
    sns.despine()


# analyze the PCA hidden state
# for i in range(n_trial_plot):
    f, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(x=H2d[i, :, 0][Y_label[i] == 0],
               y=H2d[i, :, 1][Y_label[i] == 0])
    ax.scatter(x=H2d[i, :, 0][Y_label[i] == 1],
               y=H2d[i, :, 1][Y_label[i] == 1])
    ax.plot(H2d[i, :, 0], H2d[i, :, 1], linestyle='--', color='k')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    sns.despine()
    f.tight_layout()


# compute PE around event boundaries, except for the first event


def get_tps_around_event_bounds(label, window_size=5):
    event_bonds = np.where(np.diff(label) != 0)[0]
    if len(event_bonds) == 0:
        return None
    return [range(ebi - window_size, ebi + window_size + 1) for ebi in event_bonds]


def get_data_at_tps(data_vec, tps):
    return [data_vec[tps_i] for tps_i in tps]


# get errors around event boundaries
window_size = task.event_len // 2 - 1
error_tps = []
for i in range(n_epochs_test):
    tps = get_tps_around_event_bounds(Y_label[i], window_size=window_size)
    if tps is not None:
        error_tps.extend(get_data_at_tps(error[i], tps))
error_tps = np.array(error_tps)

# make the line plot
error_tps_mu, error_tps_se = compute_stats(error_tps)
f, ax = make_line_plot_wer(
    error_tps_mu, error_tps_se, figsize=(6, 4), ylabel='|PE|', xlabel='Time',
    ylim=[-.05, 1]
)
ax.axvline(window_size, linestyle='--', color='red', label='event bond')
ax.legend()
