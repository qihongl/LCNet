'''
plot the results (averaged across subjects) from sim-poly.py
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from utils import pickle_load_dict
from stats import compute_stats

sns.set(style='white', palette='colorblind', context='poster')
model_names = ['The proposed model', 'Separate representation', 'Regular neural network']
# n_subjs = 50
# n_poly = 8
# n_epochs = 2**3
# block_size = 2**11
n_subjs = 20
n_poly = 4
# n_epochs = 2**7
# block_size = 2**6
# n_epochs = 2**4
# block_size = 2**9
# n_epochs = 2**11
# block_size = 2**2
# n_epochs = 2**2
# block_size = 2**11
# training params
exp_sum = 13
for i in np.arange(2, 13, 2):
    n_epochs, block_size = 2**i, 2**(exp_sum-i)

    # n_epochs = 2**9
    # block_size = 2**6
    dim_hidden = 128

    log_root = '../log'
    sim_name = f'poly-{n_poly}/epochs-{n_epochs}-block-{block_size}/n_hidden-{dim_hidden}'
    log_dir = os.path.join(log_root, sim_name)

    loss_raw_c, loss_raw_s, loss_raw_r = [], [], []
    loss_c, loss_s, loss_r = [], [], []
    eloss_c, eloss_s, eloss_r = [], [], []
    bceloss_c, bceloss_s, bceloss_r = [], [], []
    for subj_id in range(n_subjs):
        # extract data
        data_path = os.path.join(log_dir, f'poly-subj-{subj_id}.pkl')
        data_dict = pickle_load_dict(data_path)
        loss_c_si = data_dict['log_loss_c']
        loss_s_si = data_dict['log_loss_s']
        loss_r_si = data_dict['log_loss_r']
        eloss_c_si = data_dict['log_eloss_c']
        eloss_s_si = data_dict['log_eloss_s']
        eloss_r_si = data_dict['log_eloss_r']
        # the raw data before averaging
        loss_raw_c.append(loss_c_si)
        loss_raw_s.append(loss_s_si)
        loss_raw_r.append(loss_r_si)
        # training MSE, average across subjs
        loss_c.append(np.mean(loss_c_si, axis=1))
        loss_s.append(np.mean(loss_s_si, axis=1))
        loss_r.append(np.mean(loss_r_si, axis=1))
        # get the mean MSE for each context
        bceloss_c.append(np.mean(eloss_c_si, axis=2))
        bceloss_s.append(np.mean(eloss_s_si, axis=2))
        bceloss_r.append(np.mean(eloss_r_si, axis=2))
        # average across subjects
        eloss_c.append(np.mean(bceloss_c[subj_id], axis=1))
        eloss_s.append(np.mean(bceloss_s[subj_id], axis=1))
        eloss_r.append(np.mean(bceloss_r[subj_id], axis=1))

    # compute mu
    loss_raw_c_mu, loss_raw_c_se = compute_stats(np.array(loss_raw_c), axis=0)
    loss_raw_s_mu, loss_raw_s_se = compute_stats(np.array(loss_raw_s), axis=0)
    loss_raw_r_mu, loss_raw_r_se = compute_stats(np.array(loss_raw_r), axis=0)
    loss_c_mu, loss_c_se = compute_stats(np.array(loss_c))
    loss_s_mu, loss_s_se = compute_stats(np.array(loss_s))
    loss_r_mu, loss_r_se = compute_stats(np.array(loss_r))
    eloss_c_mu, eloss_c_se = compute_stats(np.array(eloss_c))
    eloss_s_mu, eloss_s_se = compute_stats(np.array(eloss_s))
    eloss_r_mu, eloss_r_se = compute_stats(np.array(eloss_r))
    bceloss_c_mu, bceloss_c_se = compute_stats(np.array(bceloss_c), axis=0)
    bceloss_s_mu, bceloss_s_se = compute_stats(np.array(bceloss_s), axis=0)
    bceloss_r_mu, bceloss_r_se = compute_stats(np.array(bceloss_r), axis=0)

    # pack the data
    loss_raw_mu = [loss_raw_c_mu, loss_raw_s_mu, loss_raw_r_mu]
    loss_raw_se = [loss_raw_c_se, loss_raw_s_se, loss_raw_r_se]
    loss_mu = [loss_c_mu, loss_s_mu, loss_r_mu]
    loss_se = [loss_c_se, loss_s_se, loss_r_se]
    eloss_mu = [eloss_c_mu, eloss_s_mu, eloss_r_mu]
    eloss_se = [eloss_c_se, eloss_s_se, eloss_r_se]
    bceloss_mu = [bceloss_c_mu, bceloss_s_mu, bceloss_r_mu]
    bceloss_se = [bceloss_c_se, bceloss_s_se, bceloss_r_se]

    # plot colors
    cpal = sns.color_palette(n_colors=len(model_names))
    blues = sns.color_palette('Blues', n_colors=n_poly)
    oranges = sns.color_palette('Oranges', n_colors=n_poly)
    greens = sns.color_palette('Greens', n_colors=n_poly)
    cpals = [blues, oranges, greens]

    # '''plot the data'''
    # # training error
    # f, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True)
    # for mi, mname in enumerate(model_names):
    #     ax.plot(loss_mu[mi], color=cpal[mi], label=model_names[mi])
    #     ax.errorbar(x=range(n_epochs), y=loss_mu[mi], yerr=loss_se[mi],
    #                 color=cpal[mi], alpha=1)
    # ax.axhline(0, linestyle='--', color='grey')
    # ax.set_title(f'n polys = {n_poly}, block size = {block_size}')
    # ax.legend()
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Training MSE')
    # sns.despine()
    # f.tight_layout()


    # plot validation error
    f, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True)

    for mi, mname in enumerate(model_names):
        # axes[mi].plot(bceloss_mu[mi][:, pi], color=blues[pi])
        ax.plot(eloss_mu[mi], label=model_names[mi], color=cpal[mi])
        ax.errorbar(x=range(n_epochs), y=eloss_mu[mi], yerr=eloss_se[mi],
                    color=cpal[mi], alpha=.2)

    ax.axhline(0, linestyle='--', color='grey')
    ax.set_title(f'n polys = {n_poly}, block size = {block_size}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    # ax.legend()
    sns.despine()
    f.tight_layout()


    #
    # # training error separated by epochs
    # N = 4
    # trunc = 512
    # f, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    # # f, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True)
    # for mi, (mname, ax) in enumerate(zip(model_names, axes)):
    #     for ei in range(n_epochs):
    #         mvmu_ = np.convolve(
    #             loss_raw_mu[mi][ei, :trunc], np.ones(N) / N, mode='valid')
    #         ax.plot(mvmu_[:trunc], color=cpals[mi][ei], label=f'poly {ei}')
    #     ax.set_xlabel('# samples')
    #     ax.set_title(model_names[mi])
    #     ax.set_ylabel('Training MSE')
    #     # ax.legend()
    #     sns.despine()
    # f.tight_layout()
    #
    #
    #
    # #legend
    # alphas = [.1, .4, .7, 1]
    # f, ax = plt.subplots(1, 1, figsize=(5, 5))
    # for i in range(n_poly):
    #     ax.plot([0], color='k', label=f'poly {i}', alpha=alphas[i])
    # ax.legend()
    #
    # # plot test MSE for each poly in a n-block away fashion
    # f, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    # for pi in range(n_poly):
    #     for mi, mname in enumerate(model_names):
    #         # axes[mi].plot(bceloss_mu[mi][:, pi], color=blues[pi])
    #         axes[mi].errorbar(x=range(n_epochs), y=bceloss_mu[mi][:, pi],
    #                           yerr=bceloss_se[mi][:, pi], color=cpals[mi][pi])
    #         axes[mi].set_title(mname)
    #
    # for i, ax in enumerate(axes):
    #     ax.axhline(0, linestyle='--', color='grey')
    #     ax.set_xlabel('Epochs')
    #     ax.set_ylabel('MSE')
    #     ax.set_xticklabels(range(n_epochs))
    #     ax.set_xticks(range(n_epochs))
    #     ax.set_ylim([-.015, .35])
    # sns.despine()
    # f.tight_layout()
