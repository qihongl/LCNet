import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np
import itertools
import torch
import os

from task import MixedCSW
from model import dLocalMAP as CRP
from model import CRPLSTM
from utils import to_sqpth, to_pth, to_np, to_sqnp
from copy import deepcopy
from stats import compute_stats
from vis import plot_learning_curve
from sklearn.decomposition import PCA

sns.set(style='white', palette='colorblind', context='talk')
N_CONTEXTS = 2

subj_id = 0
# for subj_id in range(10):
torch.manual_seed(subj_id)
np.random.seed(subj_id)


'''simulation params'''
# task params
event_len = 10
ctx_dim = 32
vec_dim, reptype = None, 'onehot'
n_train = 64
n_total = n_train * 2
n_meta_loops = 256
# training params
lr = 3e-3
c = .1
lmbd = 0
alpha = np.ones((ctx_dim, 2))
# rnn params
dim_hidden = 32


'''init task and model'''
mcsw = MixedCSW(
    event_len=event_len, vec_dim=vec_dim, ctx_dim=ctx_dim, reptype=reptype
)
# init model
dim_input = mcsw.vec_dim + ctx_dim * 2
dim_output = mcsw.vec_dim
model = CRPLSTM(dim_input, dim_hidden, dim_output, use_ctx=False,
    output_format='continuous')
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

'''training'''

log_acc = np.zeros((n_meta_loops, n_total, event_len - 1))
log_C = np.zeros((n_meta_loops, n_total, event_len - 1, dim_hidden))
log_c_a_ids = np.zeros((n_meta_loops, n_total),dtype=np.int)
log_c_b_ids = np.zeros((n_meta_loops, n_total),dtype=np.int)
log_loss = np.zeros(n_meta_loops)

for j in range(n_meta_loops):
    # reinit crps
    crp_a = CRP(c=c, alpha=alpha, lmbd=lmbd, context_dim=ctx_dim,
              allow_redundant_instance=True)
    crp_b = CRP(c=c, alpha=alpha, lmbd=lmbd, context_dim=ctx_dim,
              allow_redundant_instance=True)

    # reinit task
    mcsw = MixedCSW(
        event_len=event_len, vec_dim=vec_dim, ctx_dim=ctx_dim, reptype=reptype
    )
    paths, context_ids = mcsw.sample_compositional(n_train, to_pytorch=True)

    loss = 0

    # loop over all trials in this meta loop
    for i in range(n_total):
        # # use the noisy percepts to infer the underlying context
        ctx_a, ctx_b = mcsw.get_ctx_rep(context_ids[i])
        log_c_a_ids[j, i] = crp_a.stimulate(np.array(np.round(ctx_a), dtype=np.int))
        log_c_b_ids[j, i] = crp_b.stimulate(np.array(np.round(ctx_b), dtype=np.int))
        c_a, c_b = crp_a.context[log_c_a_ids[j, i]], crp_b.context[log_c_b_ids[j, i]]

        # use the ground truth context
        # c_a, c_b = mcsw.get_ctx_rep(context_ids[i])

        # loop over time
        h_t, c_t = model.get_init_states()
        loss_train = 0
        for t in range(event_len - 1):
            # combine the input and the context information
            x_t = torch.cat([paths[i, t], to_pth(c_a), to_pth(c_b)])
            y_t = paths[i, t + 1]
            # recurrent computation at time t
            [yhat, prob_a_t, v_t, h_t, c_t], cache = model(x_t.view(1, 1, -1), h_t, c_t)
            if i < n_train:
                loss_train += criterion(torch.squeeze(yhat), y_t)
            else:
                # compute loss
                loss += criterion(torch.squeeze(yhat), y_t)
            # log
            log_acc[j, i, t] = torch.argmax(
                torch.squeeze(yhat)) == torch.argmax(y_t)
            log_C[j, i, t] = to_sqnp(c_t)

        # reduce the training loss of the i-th trial
        if i < n_train:
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

    # reduce the loss of the j-th meta loop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    log_loss[j] = loss
    test_acc_j = np.mean(log_acc[j, n_train:, :])
    print('%3d | test acc = %.4f, test loss = %.4f' % (j, test_acc_j, log_loss[j]))


f, ax = plot_learning_curve(
    log_loss, ylabel='Loss', window_size=1, alpha=.3,
    final_error_winsize=10, figsize=(8, 4)
)
ax.legend()

mean_acc = np.mean(log_acc, axis=2)
f, ax = plt.subplots(1,1, figsize=(8, 4))
sns.heatmap(mean_acc, ax=ax, cmap='bone')
ax.axvline(n_train, linestyle = '--', color='red', label ='start testing')
ax.set_xlabel('Trials')
ax.set_ylabel('Meta-learning epochs')
ax.set_title('Average accuracy')

# analyze the performance of the last meta loop
n_trials_to_plot = 50
final_acc = np.mean(log_acc[-n_trials_to_plot:,:,:],axis=0)
early_acc = np.mean(log_acc[:n_trials_to_plot,:,:],axis=0)


final_acc_mu, final_acc_se = compute_stats(final_acc,axis=1)
early_acc_mu, early_acc_se = compute_stats(early_acc,axis=1)

f, ax = plt.subplots(1,1, figsize=(8,4))
ax.plot(final_acc_mu, label=f'accuracy, last {n_trials_to_plot} epochs')
ax.fill_between(range(len(final_acc_mu)), final_acc_mu-final_acc_se, final_acc_mu+final_acc_se, alpha=.2)
ax.plot(early_acc_mu, label=f'accuracy, first {n_trials_to_plot} epochs')
ax.fill_between(range(len(early_acc_mu)), early_acc_mu-early_acc_se, early_acc_mu+early_acc_se, alpha=.2)
ax.set_xlabel('Trials')
ax.set_ylabel('Accuracy')
ax.axvline(n_train, linestyle = '--', color='red', label ='start testing')
ax.axhline(.5, linestyle = '--', color='grey', label ='chance')
ax.legend()
sns.despine()


f, ax = plt.subplots(1,1, figsize=(8,4))
ax.plot(log_c_a_ids[0], label='latent cause A',color=sns.color_palette()[3])
ax.plot(log_c_b_ids[0], label='latent cause B',color=sns.color_palette()[2])
ax.axvline(n_train, linestyle = '--', color='red', label ='start testing')
ax.set_ylabel('True latent cause')
ax.set_xlabel('Trials')
ax.legend()
sns.despine()
