'''
train CRP-NN, SEM, regular networks (all feed forward)
to learn multiple polynomials with a shared component

each task/poly = a shared component + a idio compoennt
'''

import os
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from task import Polynomial
from model import CRPNN, SEM, SimpleNN
from model import dLocalMAP as ContextRep
from utils import sample_binary_array, flip_bits, pickle_save_dict, \
    to_pth, to_sqpth, to_np, to_sqnp, count_cluster_assignment, mapping_count
sns.set(style='white', palette='colorblind', context='poster')


def run_models(crpnn_, sem_, rnet_, crp_, p_i, batch_size_, learning=True):
    # random
    # p_i = np.random.choice(range(n_poly))
    npercept_i = flip_bits(percept[p_i], noise_level)

    # get the vector rep of the ongoing poly for crpnn
    c_id = crp_.stimulate(npercept_i)
    context_i = crp_.context[c_id]

    # activate the k-th sem model
    if c_id >= len(sem_.models):
        sem_.add_model()

    # sample a batch of data from the ongoing poly
    pts_id = np.random.choice(
        range(spoly.n_pts), size=batch_size_, replace=True
    )
    y = Y[p_i, pts_id]
    x = X[pts_id]
    log_crp_assignm[i, :] = np.array([p_i, c_id])

    log_loss_c_ = np.zeros(batch_size_,)
    log_loss_s_ = np.zeros(batch_size_,)
    log_loss_r_ = np.zeros(batch_size_,)
    # loop over all data from this batch
    for j in range(batch_size_):
        # CRPNN
        yhat_crp = crpnn_.forward(
            to_pth(x[j]).view(1,  -1), to_pth(context_i).view(1,  -1)
        )
        # update weights
        loss_crpnn = criterion(yhat_crp, to_pth(y[j]).view(1, 1))
        if learning:
            optimizer_crpnn.zero_grad()
            loss_crpnn.backward()
            optimizer_crpnn.step()

        # SEM
        yhat_sem = sem_.models[c_id].forward(to_pth(x[j]).view(1,  -1))
        # update weights
        loss_sem = criterion(yhat_sem, to_pth(y[j]).view(1, 1))
        if learning:
            sem_.optims[c_id].zero_grad()
            loss_sem.backward()
            sem_.optims[c_id].step()

        # regular net
        xp = np.concatenate([np.reshape(x[j], (1,)), npercept_i])
        yhat_r = rnet_.forward(to_pth(xp).view(1,  -1))
        # update weights
        loss_r = criterion(yhat_r, to_pth(y[j]).view(1, 1))
        if learning:
            optimizer_rnet.zero_grad()
            loss_r.backward()
            optimizer_rnet.step()

        # log info
        log_loss_c_[j] = to_np(loss_crpnn)
        log_loss_s_[j] = to_np(loss_sem)
        log_loss_r_[j] = to_np(loss_r)
    return log_loss_c_, log_loss_s_, log_loss_r_


savedata = True

'''set simulation parameters'''
# env/task params
n_poly = 4
noise_level = .1
n_term_shared = 2
n_term_idio = 2
degree_shared = 2
degree_idio = 3
xrange = 1

# agent params
dim_percept = 32
dim_hidden = 128
dim_context = 32
dim_input = 1
dim_output = 1
c = .1
alpha = np.ones((dim_percept, 2))


# training params
exp_sum = 13
for i in np.arange(2, 13, 2):
    n_epochs, block_size = 2**i, 2**(exp_sum-i)
    # n_epochs = 2**11
    # block_size = 2**2
    # n_epochs = 2**2
    # block_size = 2**11
    # n_epochs = 2**4
    # block_size = 2**9
    # n_epochs = 2**9
    # block_size = 2**6
    # n_epochs = 2**2
    # block_size = 2**11

    block_size_eval = 100
    lr = 1e-5
    print(f'n epochs x block size = {n_epochs} x {block_size}')

    # number of subjects
    subj_id = 0
    n_subjs = 20
    for subj_id in np.arange(0, n_subjs):
        print(subj_id)
        np.random.seed(subj_id)
        torch.manual_seed(subj_id)

        '''generate data'''
        # sample a different perception pattern for each polynomial
        percept = sample_binary_array(dim_percept, n_poly)
        # get the shared poly
        spoly = Polynomial(
            n_terms=n_term_shared, max_degree=degree_shared, xrange=xrange
        )
        # get the idiosyncratic poly
        poly = [
            Polynomial(n_terms=n_term_idio, max_degree=degree_idio, xrange=xrange)
            for _ in range(n_poly)
        ]

        # sample from the shared components
        X, sy = spoly.sample()
        # sample from idiosyncratic components
        ix, iy = np.zeros((n_poly, spoly.n_pts)), np.zeros((n_poly, spoly.n_pts))
        for i in range(n_poly):
            ix[i, :], iy[i, :] = poly[i].sample()
        # sum to get y
        Y = iy + np.tile(sy, (n_poly, 1))

        '''init CRP-NN'''
        # x = torch.zeros(1, 1)
        # c_k = torch.randn(1, 1, dim_context)
        crp = ContextRep(c=c, alpha=alpha, lmbd=0, context_dim=dim_context)
        crpnn = CRPNN(dim_input, dim_hidden, dim_output, dim_context)
        optimizer_crpnn = torch.optim.Adam(crpnn.parameters(), lr=lr)
        criterion = nn.MSELoss()

        '''init SEM'''
        sem = SEM(dim_input, dim_hidden, dim_output, lr)

        '''init a regular nn'''
        rnet = SimpleNN(dim_input + dim_percept, dim_hidden, dim_output)
        optimizer_rnet = torch.optim.Adam(rnet.parameters(), lr=lr)

        '''train the model'''
        log_loss_c = np.zeros((n_epochs, block_size))
        log_loss_s = np.zeros((n_epochs, block_size))
        log_loss_r = np.zeros((n_epochs, block_size))
        log_eloss_c = np.zeros((n_epochs, n_poly, block_size_eval))
        log_eloss_s = np.zeros((n_epochs, n_poly, block_size_eval))
        log_eloss_r = np.zeros((n_epochs, n_poly, block_size_eval))
        log_n_contexts = np.zeros(n_epochs,)
        log_crp_assignm = np.zeros((n_epochs, 2))
        # i, j = 0, 0
        for i in range(n_epochs):
            # choose the id for the ongoing poly/task
            p_i = i % n_poly
            # random
            # p_i = np.random.choice(range(n_poly))
            log_loss_c[i, :], log_loss_s[i, :], log_loss_r[i, :] = run_models(
                crpnn, sem, rnet, crp, p_i, block_size, learning=True)

            for ep_i in range(n_poly):
                eval_results = run_models(
                    deepcopy(crpnn), deepcopy(sem), deepcopy(rnet), deepcopy(crp),
                    ep_i, block_size_eval, learning=False
                )
                log_eloss_c[i, ep_i, :], log_eloss_s[i, ep_i, :], \
                    log_eloss_r[i, ep_i, :] = eval_results

            print(
                'epoch %4d | loss: crpnn: %.4f, sem: %.4f reg: %.4f' % (
                    i, np.mean(log_loss_c[i]), np.mean(log_loss_s[i]), np.mean(log_loss_r[i]))
            )

        '''save data'''
        if savedata:
            # construct data path
            log_root = '../log'
            sim_name = f'poly-{n_poly}/epochs-{n_epochs}-block-{block_size}/n_hidden-{dim_hidden}'
            log_dir = os.path.join(log_root, sim_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # save data
            data_dict = {
                'log_loss_c': log_loss_c, 'log_loss_s': log_loss_s, 'log_loss_r': log_loss_r,
                'log_eloss_c': log_eloss_c, 'log_eloss_s': log_eloss_s, 'log_eloss_r': log_eloss_r,
            }
            data_path = os.path.join(log_dir, f'poly-subj-{subj_id}.pkl')
            pickle_save_dict(data_dict, data_path)

        '''plot the data'''
        f, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True)
        # win_size = 10
        # log_loss_c_mu = np.convolve(
        #     np.mean(log_loss_c, axis=1), np.ones(win_size) / win_size, mode='valid')
        # log_loss_s_mu = np.convolve(
        #     np.mean(log_loss_s, axis=1), np.ones(win_size) / win_size, mode='valid')
        # log_loss_r_mu = np.convolve(
        #     np.mean(log_loss_r, axis=1), np.ones(win_size) / win_size, mode='valid')
        # ax.plot(log_loss_c_mu, label='crpnn')
        # ax.plot(log_loss_s_mu, label='sep net')
        # ax.plot(log_loss_r_mu, label='regular net')
        ax.plot(np.mean(log_loss_c, axis=1), label='the proposed model')
        ax.plot(np.mean(log_loss_s, axis=1), label='sep net')
        ax.plot(np.mean(log_loss_r, axis=1), label='regular net')

        ax.axhline(0, linestyle='--', color='grey')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE')
        ax.legend()
        sns.despine()
        f.tight_layout()

        # f, ax = plt.subplots(1, 1, figsize=(6, 4))
        # ax.plot(log_n_contexts)

        f, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True)
        # compute the test MSE for each poly
        log_eloss_s_context_mu = np.mean(log_eloss_s, axis=2)
        log_eloss_c_context_mu = np.mean(log_eloss_c, axis=2)
        log_eloss_r_context_mu = np.mean(log_eloss_r, axis=2)
        # average all polys
        log_eloss_s_mumu = np.mean(log_eloss_s_context_mu, axis=1)
        log_eloss_c_mumu = np.mean(log_eloss_c_context_mu, axis=1)
        log_eloss_r_mumu = np.mean(log_eloss_r_context_mu, axis=1)
        # plot the data
        ax.plot(log_eloss_c_mumu, label='crpnn')
        ax.plot(log_eloss_s_mumu, label='sep net')
        ax.plot(log_eloss_r_mumu, label='regular net')

        ax.axhline(0, linestyle='--', color='grey')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE')
        ax.legend()
        sns.despine()
        f.tight_layout()

        # plot the polys
        f, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        axes[0].plot(X, sy, color='k', ls='--')
        for i in range(n_poly):
            axes[0].plot(X, iy[i, :])
            axes[1].plot(X, Y[i, :])
        for ax in axes:
            ax.set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title('Shared component \nand idiosyncratic components')
        axes[0].legend(['shared component'])
        axes[1].set_title('Target functions')
        sns.despine()
        f.tight_layout()

        # plot the individual/shared components and the model fits
        f, axes = plt.subplots(n_poly // 2, 2, figsize=(12, 10))

        for p_i, ax in zip(range(n_poly), axes.reshape(-1)):
            c_id = crp.stimulate(percept[p_i])
            context_i = crp.context[c_id]

            yhat_c = [to_sqnp(crpnn.forward(
                to_pth(x_j).view(1,  -1), to_pth(context_i).view(1,  -1)
            ))for x_j in X]
            yhat_s = [to_sqnp(sem.models[c_id].forward(to_pth(x_j).view(1,  -1)))
                      for x_j in X]

            ax.plot(X, yhat_c, label='model prediction')
            # ax.plot(X, yhat_s, label='yhat sep net')
            ax.plot(X, Y[p_i, ], label='truth', color='k', linestyle='--')
            ax.set_title(f'Polynomial {p_i}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
        f.tight_layout()
        sns.despine()

        empty_context = torch.zeros(np.shape(context_i))
        f, ax = plt.subplots(1, 1, figsize=(6, 5))
        yhat_shared = [to_sqnp(crpnn.forward(
            to_pth(x_j).view(1,  -1), to_pth(empty_context).view(1,  -1)
        ))for x_j in X]
        ax.plot(X, yhat_shared, label='LCI-lesioned model')
        ax.plot(X, sy, label='shared component', color='k', linestyle='--')
        ax.set_title(f'Model fit')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        sns.despine()
