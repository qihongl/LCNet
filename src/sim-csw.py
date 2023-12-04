import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import pandas as pd
import numpy as np
import itertools
import torch
import os
from task import CSW
from model import ShortCut
from model import TabularShortCutIS
from model import SimplePETracker
from model import SimpleContext
from model import CGRU as Agent

from utils import to_sqpth, to_pth, to_np, to_sqnp, cosine_sim, stable_softmax, is_even
from copy import deepcopy
from stats import compute_stats, purity_score, ber_kl_div
from vis import plot_learning_curve
from sklearn.decomposition import PCA
from scipy.special import kl_div

N_CONTEXTS = 2
sns.set(style='white', palette='colorblind', context='talk')
cpal = sns.color_palette()
gb = [cpal[2], cpal[0]]

'''simulation params'''
# task params
event_len_hdim = 16 # event length just to make the input more high dim
event_len = 7 # actual event len
reptype = 'normal'
block_size = 100
condition = 'blocked'
# condition = 'interleaved'

# rnn params
lr = 4e-3
dim_hidden = 128
context_dim = 128

# SC params
pseudo_count = 1
stickiness = 32
concentration = .5
beta = .05
ctx_wt = .6
dropout_rate = 0.1
pe_threshold = .8
buffer_size = 2

n_subjs = 5
subj_id = 0
seed_start = 0

time_tick_labels = [f'{x}' for x in  np.arange(event_len-1)]
time_tick_labels[0] = '0\nCIS'
time_tick_labels[1] = '1\n(50-50)'

n_se = 3

# prealloc m x n matrix
nan_mn = lambda m, n: np.full((m,n), np.nan)
nan_mnk = lambda m, n, k: [nan_mn(m,n) for _ in range(k)]

log_kld_b, log_kld_nb,log_mse_b, log_mse_nb = nan_mnk(n_subjs, event_len-1, 4)
log_zkld_b, log_zkld_nb,log_zmse_b, log_zmse_nb = nan_mnk(n_subjs, event_len-1, 4)
log_pe_b, log_pe_nb, log_zpe_b, log_zpe_nb = nan_mnk(n_subjs, event_len-1, 4)
log_nctx = np.zeros(n_subjs)
log_purity_g = nan_mn(n_subjs, 2)
log_purity_fi_g = nan_mn(n_subjs, 2)
log_purity_perm_g = nan_mn(n_subjs, 1)
log_acc_sc_g = [None for _ in range(n_subjs)]

log_acc_g = [None for _ in range(n_subjs)]
log_ctx_g = [None for _ in range(n_subjs)]
log_loss_g = [None for _ in range(n_subjs)]
log_sc_time_g = [None for _ in range(n_subjs)]
# log_data_count_g = [None for _ in range(n_subjs)]
loc_fi_g = [None for _ in range(n_subjs)]

for subj_id in range(n_subjs):
    torch.manual_seed(subj_id + seed_start)
    np.random.seed(subj_id + seed_start)

    '''init task and model'''
    # task
    csw = CSW(event_len=event_len_hdim, reptype=reptype)
    event_bounds = np.arange(block_size, block_size * (csw.n_blocks_total+1), block_size)
    # init model
    dim_input = csw.vec_dim
    dim_output = 1

    simple_context = SimpleContext(
        context_dim, concentration=concentration, pseudo_count=pseudo_count, stickiness=stickiness)
    model = Agent(
        dim_input, dim_hidden, dim_output,
        ctx_wt=ctx_wt, context_dim=context_dim,
        sigmoid_output=True, dropout_rate=dropout_rate,
    )

    pet = SimplePETracker(size=1200)
    acct = SimplePETracker(size=1200)
    short_cut = TabularShortCutIS(buffer_size=buffer_size)

    # init optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    '''training'''
    ctx_ids, _, node_seqs, vec_seqs, vec_seqs_wctx = csw.sample_wctx2(
        block_size, to_torch=True,condition=condition)
    ctx_ids_extended = np.tile(ctx_ids, (event_len-1, 1))
    # shift the 1st time point -- infer the prev context at zero is reasonable
    ctx_ids_extended[0, 1:] = ctx_ids_extended[0, :-1]

    log_acc = np.zeros((csw.n_total_trials, event_len - 1))
    log_yhat = np.zeros((csw.n_total_trials, event_len - 1))
    log_ctx = np.full((csw.n_total_trials, event_len - 1), np.nan)
    log_ctx_fi = np.full((csw.n_total_trials, event_len - 1), np.nan)
    log_ctx_sc = np.full((csw.n_total_trials, event_len - 1), np.nan)
    log_ctx_match = np.full((csw.n_total_trials, event_len - 1), np.nan)
    log_mpe = np.full((csw.n_total_trials, event_len - 1), np.nan)
    log_lik = np.full((csw.n_total_trials, event_len - 1, csw.vec_dim), np.nan)
    log_pos = np.full((csw.n_total_trials, event_len - 1, csw.vec_dim), np.nan)
    log_loss_all = np.full((csw.n_total_trials, event_len-1), np.nan)
    log_kl_all = np.full((csw.n_total_trials, event_len-1), np.nan)
    log_z_all = np.zeros((csw.n_total_trials, event_len-1))
    log_pe_all = np.zeros((csw.n_total_trials, event_len-1))
    log_pe_sc = np.zeros((csw.n_total_trials, event_len-1))
    log_pet_acc = np.full((csw.n_total_trials, ), np.nan)
    loc_fi = np.zeros((csw.n_total_trials, event_len-1))
    loc_peak = np.full((csw.n_total_trials, event_len-1), np.nan)

    H = np.zeros((csw.n_total_trials, event_len-1, dim_hidden))
    X = np.zeros((csw.n_total_trials, event_len-1, dim_input))

    # init cluster assignment
    c_it, ctx_it = simple_context.add_new_context()
    c_fi = c_it
    t_skip = []
    i,t=0,0
    pe_peaked = False
    more_fi = 0
    prev_ctx = 0

    for i in range(csw.n_total_trials):

        h_t = model.get_zero_states()
        loss = 0
        for t in range(event_len - 1):
            # remove recurrence
            h_t = model.get_zero_states()

            # prep for forward prop
            x_t = vec_seqs_wctx[i, t].view(1, 1, -1)
            y_t_1d = ctx_ids[i] if t == 0 else is_even(node_seqs[i, t-1])

            '''info for context inference'''
            if t == 0:
                x_t_sc = to_sqnp(x_t)
            else:
                x_t_sc = to_sqnp(x_t) * .8 + x_t_sc * .2
            X[i,t] = x_t_sc

            # recurrent computation at time t
            [yhat, h_t], cache = model(to_pth(x_t_sc), h_t, to_pth(simple_context.context[int(c_it)]))

            # compute loss
            z = 2 * torch.abs(yhat - .5)
            loss_it = criterion(torch.squeeze(yhat), to_pth(y_t_1d))
            loss += loss_it

            # log info
            log_acc[i, t] = bool(abs(yhat - y_t_1d) < .5)
            log_yhat[i, t] = to_sqnp(yhat)
            log_z_all[i,t] = to_sqnp(z)
            log_loss_all[i, t] = to_np(loss_it)
            log_kl_all[i,t] = ber_kl_div(y_t_1d, to_sqnp(yhat))

            '''calculate PE'''
            # compute the PE at time t
            if i <= 1:
                log_pe_all[i,t] = log_loss_all[i, t] * log_z_all[i, t]
            else:
                denom = np.nanmean(log_loss_all[:i, t])
                log_pe_all[i,t] = log_loss_all[i, t] / denom
            pet.record(t, log_pe_all[i,t])
            # detemine if there is a PE peak
            loc_peak[i,t] = pet.get_recent_pe(t) > pet.get_mean(t) + pet.get_std(t) * pe_threshold

            '''full inference except for the 50-50 node'''
            if t != 1:
                likelihood = model.try_all_contexts(y_t_1d, to_pth(x_t_sc), h_t, to_pth(simple_context.context))
                likelihood_ = stable_softmax(likelihood, beta=beta)
                posterior = simple_context.compute_posterior(likelihood_, verbose=False)
                c_fi = simple_context.assign_context(posterior, verbose=False)
            else:
                c_fi = log_ctx_fi[i, 0]
            log_ctx_fi[i, t] = c_fi


            '''short cut inference'''

            use_full_inf = short_cut.is_novel(x_t_sc)
            # use_full_inf = short_cut.is_novel(x_t_sc) or short_cut.n_associates(x_t_sc) > 1
            prev_ctx_sc_it, ctx_sc_it = short_cut.infer_context(x_t_sc)
            log_ctx_sc[i,t] = ctx_sc_it+1 if ctx_sc_it is not None else np.nan
            # record whether full inf match with short cut
            log_ctx_match[i, t] = log_ctx_sc[i,t] == log_ctx_fi[i, t]
            # turn off EM
            use_full_inf = 1

            '''full inf vs. short cut '''
            # acc_close_to_chance = acct.get_mean(t) is not None and acct.get_mean(t) < .6
            # if acc at time t is bad or use full inf, then use full inf
            # if use_full_inf or acc_close_to_chance:
            if use_full_inf:
                c_it = log_ctx_fi[i, t]
                # TODO remove this if not? why is it here?
                # if not acc_close_to_chance:
                loc_fi[i, t] = 1
            else:
                c_it = log_ctx_sc[i, t]

            # prev_ctx = c_it

            '''whether to update the shortcut buffer'''
            # add data
            short_cut.add_data(x_t_sc, to_pth(log_ctx_fi[i, t])-1, update=use_full_inf or loc_peak[i,t])
            mu_acc_i = np.mean(log_acc[i])

            # log
            log_ctx[i, t] = c_it

        # update weights
        if i < block_size*4:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

    # log group level data
    log_purity_g[subj_id][0] = purity_score(ctx_ids_extended[:,:-block_size], log_ctx.T[:,:-block_size])
    log_purity_g[subj_id][1] = purity_score(ctx_ids_extended[:,-block_size:], log_ctx.T[:,-block_size:])
    log_purity_fi_g[subj_id][0] = purity_score(ctx_ids_extended[:,:-block_size], log_ctx_fi.T[:,:-block_size])
    log_purity_fi_g[subj_id][1] = purity_score(ctx_ids_extended[:,-block_size:], log_ctx_fi.T[:,-block_size:])
    log_purity_perm_g[subj_id] = purity_score(ctx_ids_extended, log_ctx.T[:, np.random.permutation(csw.n_total_trials)])

    log_acc_g[subj_id] = np.mean(log_acc[:, 2:],axis=1)
    log_ctx_g[subj_id] = log_ctx
    log_loss_g[subj_id] = log_loss_all
    log_nctx[subj_id] = simple_context.n_context

    '''plot the data'''
    f, ax = plt.subplots(1, 1, figsize=(12,4))
    ax.plot(np.nanmean(log_ctx, axis=1), label = 'inferred context', alpha = .5)
    # ax.plot(log_ctx, label = 'inferred context', alpha = .5)
    if block_size > 1:
        ax.plot(ctx_ids, label = 'true context', alpha = .5)
    ax.set_xlabel('Time')
    ax.set_ylabel('context id')
    ax.legend()
    sns.despine()

    # show accuracy over time
    f, ax = plot_learning_curve(
        np.mean(log_acc[:, 2:],axis=1), ylabel='Accuracy', window_size=1, color='k',
        final_error_winsize=10, figsize=(12, 5)
    )
    len_learn_phase = csw.n_blocks_total * block_size
    if condition == 'blocked':
        for i, b in enumerate(event_bounds):
            # sp_legend = 'switch point' if i == 0 else "_nolegend_"
            # ax.axvline(b, linestyle = '--', color='grey', label =sp_legend)
            ax.fill_between(np.arange(block_size*i, block_size*(i+1), .01), 1, alpha=.2, color=gb[i%2])
    else:
        # ax.axvline(block_size * 5, linestyle = '--', color='grey', label = sp_legend)
        ax.fill_between(np.arange(0, len_learn_phase, .01), 1, alpha=.2, color=np.array(gb).mean(axis=0))
    ax.fill_between(np.arange(len_learn_phase, csw.n_total_trials, .01), 1, alpha=.2, color=cpal[3])
    ax.axhline(.5, linestyle = '--', color='red', label ='chance')
    ax.set_ylim([0, 1.05])
    ax.set_title('Accuracy, excluding the unpredictable time point')
    ax.set_xlabel('Trials')
    ax.legend()

    def get_error_bvnb(log_error_t_, event_bounds, eb_win = 5):
        def remove_random_phase_data(log_error_t):
            return log_error_t[:-block_size]

        # log_error_t_ = log_kl_all[:,1]
        # remove the last random phase data
        log_error_t = remove_random_phase_data(log_error_t_)
        # remove the last event bound -- the last phase is random
        ebts = event_bounds[:-1]
        # make a boundary loc mask
        err_b_mask = np.zeros_like(log_error_t, dtype=bool)
        for eb_t in ebts:
            err_b_mask[eb_t:eb_t+eb_win] = True
        # get error at boundaries
        err_b = np.nanmean(log_error_t[err_b_mask])
        # get error at non boundaries
        err_nb = np.nanmean(log_error_t[~err_b_mask])
        return err_b, err_nb

    def get_error_all_time(logg_error):
        # err_t0 = np.mean(logg_error[:,0])
        # err_t0_med = np.percentile(logg_error[block_size*2:,0], 95)
        # err_t0_max = np.percentile(logg_error[:,0], 95)
        err_b, err_nb = np.zeros(event_len-1,),np.zeros(event_len-1,)
        for t_i, t in enumerate(np.arange(0,event_len-1)):
            err_b[t_i], err_nb[t_i] = get_error_bvnb(logg_error[:,t], event_bounds)
        return err_b, err_nb

    # log_kld_b[subj_id], log_kld_nb[subj_id] = get_error_all_time(log_kl_all)
    # log_mse_b[subj_id], log_mse_nb[subj_id] = get_error_all_time(log_loss_all)
    log_kld_b[subj_id], log_kld_nb[subj_id] = get_error_all_time(log_kl_all)
    log_mse_b[subj_id], log_mse_nb[subj_id] = get_error_all_time(log_loss_all)
    log_zkld_b[subj_id], log_zkld_nb[subj_id] = get_error_all_time(log_kl_all * log_z_all)
    log_zmse_b[subj_id], log_zmse_nb[subj_id] = get_error_all_time(log_loss_all * log_z_all)
    log_pe_b[subj_id], log_pe_nb[subj_id] = get_error_all_time(log_pe_all)
    log_zpe_b[subj_id], log_zpe_nb[subj_id] = get_error_all_time(log_pe_all * log_z_all)


    f, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.imshow(log_ctx.T,aspect='auto',cmap = 'viridis', interpolation='none')
    if condition == 'blocked':
        for b in event_bounds:
            ax.axvline(b, ls='--',color='red')
    ax.set_title('Context assignment (each color is a context); purity = %.2f, %.2f' % (log_purity_g[subj_id][0], log_purity_g[subj_id][1]))
    ax.set_ylabel('Time')
    ax.set_yticks(np.arange(event_len-1)+.5)
    ax.set_yticklabels(time_tick_labels, rotation=0)

    f, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.imshow(log_ctx_sc.T,aspect='auto',cmap = 'viridis', interpolation='none')
    if condition == 'blocked':
        for b in event_bounds:
            ax.axvline(b, ls='--',color='red')
    ax.set_title('Short cut (each color is a context); purity = %.2f, %.2f' % (log_purity_g[subj_id][0], log_purity_g[subj_id][1]))
    ax.set_ylabel('Time')
    ax.set_yticks(np.arange(event_len-1)+.5)
    ax.set_yticklabels(time_tick_labels, rotation=0)

    f, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.imshow(log_ctx_fi.T,aspect='auto',cmap = 'viridis', interpolation='none')
    if condition == 'blocked':
        for b in event_bounds:
            ax.axvline(b, ls='--',color='red')
    ax.set_title('Full inference (each color is a context); purity = %.2f, %.2f' % (log_purity_g[subj_id][0], log_purity_g[subj_id][1]))
    ax.set_ylabel('Time')
    ax.set_yticks(np.arange(event_len-1)+.5)
    ax.set_yticklabels(time_tick_labels, rotation=0)

    loc_fi_g[subj_id] = loc_fi.T
    f, ax = plt.subplots(1, 1, figsize=(13, 5))
    sns.heatmap(loc_fi_g[subj_id])
    if condition == 'blocked':
        for b in event_bounds:
            ax.axvline(b, ls='--',color='red', alpha=.5)
    ax.set_title('Full inferences')
    ax.set_ylabel('Time')
    ax.set_yticks(np.arange(event_len-1)+.5)
    ax.set_yticklabels(time_tick_labels, rotation=0)

    f, ax = plt.subplots(1, 1, figsize=(13, 5))
    sns.heatmap(loc_peak.T)
    if condition == 'blocked':
        for b in event_bounds:
            ax.axvline(b, ls='--',color='red', alpha=.5)
    ax.set_title('PE peaks')
    ax.set_ylabel('Time')
    ax.set_yticks(np.arange(event_len-1)+.5)
    ax.set_yticklabels(time_tick_labels, rotation=0)


    f, ax = plt.subplots(1, 1, figsize=(13, 5))
    sns.heatmap(log_loss_all.T,cmap = 'viridis')
    if condition == 'blocked':
        for b in event_bounds:
            ax.axvline(b, ls='--',color='red', alpha=.5)
    ax.set_title('MSE')
    ax.set_ylabel('Time')
    ax.set_yticks(np.arange(event_len-1)+.5)
    ax.set_yticklabels(time_tick_labels, rotation=0)



    f, ax = plt.subplots(1, 1, figsize=(13, 5))
    sns.heatmap(log_loss_all.T* log_z_all.T,cmap = 'viridis')
    if condition == 'blocked':
        for b in event_bounds:
            ax.axvline(b, ls='--',color='red', alpha=.5)
    ax.set_title('MSE, normalized')
    ax.set_ylabel('Time')
    ax.set_yticks(np.arange(event_len-1)+.5)
    ax.set_yticklabels(time_tick_labels, rotation=0)



'''group level analysis '''
def plot_learning_curve(
    data, ylabel='MSE', window_size=1, alpha=.3, final_error_winsize=100, color=cpal[0], figsize=(8, 4)
):
    f, ax = plt.subplots(1, 1, figsize=figsize)
    if window_size > 1:
        smoothed_data = moving_average(data, window_size)
        smoothed_data_lab = f'smoothed (window size = {window_size})'
        ax.plot(smoothed_data, color=color, label=smoothed_data_lab)
        ax.plot(data, alpha=alpha, color=color, label='raw data')
        ax.legend()
    else:
        ax.plot(data, color=color, label = ylabel)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(ylabel)
    # compute average final error from the last 100 epochs
    final_ymean = np.mean(data[-min(final_error_winsize, len(data)):])
    # ax.set_title('Final %s = %.4f' % (ylabel, final_ymean))
    # sns.despine()
    f.tight_layout()
    return f, ax

log_acc_g_mu, log_acc_g_se = compute_stats(log_acc_g,axis=0, n_se=n_se)
N = 5
log_acc_g_mu = np.convolve(log_acc_g_mu, np.ones(N)/N, mode='valid')
log_acc_g_se = np.convolve(log_acc_g_se, np.ones(N)/N, mode='valid')

f, ax = plot_learning_curve(
    log_acc_g_mu, ylabel='Accuracy', alpha=.3, final_error_winsize=10, figsize=(8, 4), color='k'
)
ax.fill_between(np.arange(len(log_acc_g_mu)), log_acc_g_mu+log_acc_g_se, log_acc_g_mu-log_acc_g_se, alpha=.2, color='k')
len_learn_phase = csw.n_blocks_total * block_size
if condition == 'blocked':
    for i, b in enumerate(np.arange(block_size, block_size * 5, block_size)):
        # sp_legend = 'switch point' if i == 0 else "_nolegend_"
        # ax.axvline(b, linestyle = '--', color='grey', label =sp_legend)
        ax.fill_between(np.arange(block_size*i, block_size*(i+1), .01), 1, alpha=.2, color=gb[i%2])
else:
    # ax.axvline(block_size * 5, linestyle = '--', color='grey', label = sp_legend)
    ax.fill_between(np.arange(0, len_learn_phase, .01), 1, alpha=.2, color=np.array(gb).mean(axis=0))
ax.fill_between(np.arange(len_learn_phase, csw.n_total_trials, .01), 1, alpha=.2, color=cpal[3])
ax.axhline(.5, color='grey', label ='chance')
ax.set_ylim([0, 1.05])
# ax.set_title('Accuracy, excluding the unpredictable time point')
ax.set_xlabel('Trials')
ax.legend()


purity_df_cond = ['learning'] * n_subjs + ['test'] * n_subjs
purity_df_val = list(log_purity_g.T.reshape(-1))
purity_learn, purity_test = np.mean(log_purity_g,axis=0)
purity_df = pd.DataFrame(zip(purity_df_cond, purity_df_val), columns=['phase', 'purity'])

f, ax = plt.subplots(1, 1, figsize=(3, 3.5))
sns.boxplot(data=purity_df, y = 'purity', x = 'phase', orient='v', ax=ax)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
sns.swarmplot(data=purity_df, y = 'purity', x = 'phase', orient='v', ax=ax, size=10)
ax.set_ylim([.45, 1.05])
ax.set_ylabel('Purity')
ax.set_title('%.3f / %.3f' % (purity_learn, purity_test))
ax.axhline(.5, label = 'chance', ls='--', color='grey')
ax.legend()
sns.despine()


purity_df_cond = ['learning'] * n_subjs + ['test'] * n_subjs
purity_df_val = list(log_purity_fi_g.T.reshape(-1))
purity_learn, purity_test = np.mean(log_purity_fi_g,axis=0)
purity_df = pd.DataFrame(zip(purity_df_cond, purity_df_val), columns=['phase', 'purity'])

f, ax = plt.subplots(1, 1, figsize=(3, 3.5))
sns.boxplot(data=purity_df, y = 'purity', x = 'phase', orient='v', ax=ax)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
sns.swarmplot(data=purity_df, y = 'purity', x = 'phase', orient='v', ax=ax, size=10)
ax.set_ylim([.45, 1.05])
ax.set_ylabel('Purity, full inference')
ax.set_title('%.3f / %.3f' % (purity_learn, purity_test))
ax.axhline(.5, label = 'chance', ls='--', color='grey')
ax.legend()
sns.despine()



f, ax = plt.subplots(1,1, figsize=(7,4))
nlcs_each_subj = [np.max(x[:block_size*4]) for x in log_ctx_g]
# sns.kdeplot(nlcs_each_subj, ax=ax)
sns.histplot(nlcs_each_subj, ax=ax, discrete=True, stat='probability')
xticks = [int(x) for x in np.arange(1, np.max(log_ctx_g)+1)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ax.set_xlim([None, 16])
ax.set_xlabel('# latent causes inferred')
ax.set_title('median = %.2f' % np.median(nlcs_each_subj))
sns.despine()

f, ax = plt.subplots(1, 1, figsize=(13, 5))
sns.heatmap(np.mean(np.array(log_ctx_g),axis=0).T,cmap = 'viridis')
if condition == 'blocked':
    for b in event_bounds:
        ax.axvline(b, ls='--',color='red', alpha=.5)
ax.set_title('Context assignment (each color is a context)')
ax.set_ylabel('Time')
ax.set_yticks(np.arange(event_len-1)+.5)
ax.set_yticklabels(time_tick_labels, rotation=0)

p_full_inference = np.mean(loc_fi_g,axis=0)
n_full_infs = [np.sum(loc_fi_gi) for loc_fi_gi in loc_fi_g]
n_full_infs_mu, n_full_infs_se = compute_stats(n_full_infs, n_se=n_se)
f, ax = plt.subplots(1, 1, figsize=(13, 5))
sns.heatmap(p_full_inference, cmap='bone')
if condition == 'blocked':
    for b in event_bounds:
        ax.axvline(b, ls='--',color='red', alpha=.5)
ax.set_title(f'p(Full inferences)')
ax.set_ylabel('Time')
ax.set_yticks(np.arange(event_len-1)+.5)
ax.set_yticklabels(time_tick_labels, rotation=0)



f, ax = plt.subplots(1, 1, figsize=(13, 5))
sns.heatmap(np.mean(log_loss_g,axis=0).T,cmap = 'viridis')
if condition == 'blocked':
    for b in event_bounds:
        ax.axvline(b, ls='--',color='red', alpha=.5)
ax.set_title(f'MSE, averaged across (n = {n_subjs}) subjs ')
ax.set_ylabel('Time')
ax.set_yticks(np.arange(event_len-1)+.5)
ax.set_yticklabels(time_tick_labels, rotation=0)


log_acc_g_mu, log_acc_g_se = compute_stats(p_full_inference,axis=0, n_se=n_se)
N = 5
log_acc_g_mu = np.convolve(log_acc_g_mu, np.ones(N)/N, mode='valid')
log_acc_g_se = np.convolve(log_acc_g_se, np.ones(N)/N, mode='valid')
ylim_max = np.max(log_acc_g_mu+log_acc_g_se)*1.1

# f, ax = plot_learning_curve(
#     log_acc_g_mu, ylabel='Accuracy', alpha=.3, final_error_winsize=10, figsize=(8, 4), color='k'
# )
f, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(log_acc_g_mu, alpha=.8, color='k')
ax.fill_between(np.arange(len(log_acc_g_mu)), log_acc_g_mu, 0, alpha=.5, color='k')
len_learn_phase = csw.n_blocks_total * block_size
if condition == 'blocked':
    for i, b in enumerate(np.arange(block_size, block_size * 5, block_size)):
        # sp_legend = 'switch point' if i == 0 else "_nolegend_"
        # ax.axvline(b, linestyle = '--', color='grey', label =sp_legend)
        ax.fill_between(np.arange(block_size*i, block_size*(i+1), .01), 1, alpha=.2, color=gb[i%2])
else:
    # ax.axvline(block_size * 5, linestyle = '--', color='grey', label = sp_legend)
    ax.fill_between(np.arange(0, len_learn_phase, .01), ylim_max, alpha=.2, color=np.array(gb).mean(axis=0))

ax.fill_between(np.arange(len_learn_phase, csw.n_total_trials, .01), 1, alpha=.2, color=cpal[3])
ax.set_ylim([-.02, .6])
ax.set_ylabel('p(Full inferences)')
ax.set_xlabel('Trials')
# ax.legend()
# 1 - n_full_infs_mu / np.shape(np.mean(p_full_inference,axis=0))

f, ax = plt.subplots(1, 1, figsize=(13, 5))
ax.plot(np.mean(p_full_inference,axis=0))
if condition == 'blocked':
    for b in event_bounds:
        ax.axvline(b, ls='--',color='red', alpha=.5)
ax.set_title(f'when full inference happens over trials (mean # full inferences per subj = {n_full_infs_mu})')
ax.set_ylabel('p(Full inferences)')
ax.set_xlabel('Trial id')
f.tight_layout()
sns.despine()
