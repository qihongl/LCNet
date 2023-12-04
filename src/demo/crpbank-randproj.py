# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white', palette='colorblind', context='poster')

n_dim = 16
n_context = 2
bank_size = 3
percept_noise = .2
n_tps_per_context = 5
T = n_tps_per_context * n_context
contexts = [np.random.normal(loc=0,scale=1,size=n_dim) for i in range(n_context)]

# a random projection for each CRP
rand_proj = [np.random.rand(n_dim,n_dim) for i in range(bank_size)]


# create percept t for context i = context_i + noise_it
percepts = []
for i in range(n_context):
    for t in range(n_tps_per_context):
        noise_it = np.random.normal(loc=0,scale=percept_noise,size=n_dim)
        percepts.append(contexts[i] + noise_it)
percepts = np.array(percepts)


# plot the inter-percept representational similarity matrix
p2p_cov = np.corrcoef(percepts)

f, ax = plt.subplots(1,1, figsize=(5,4))
sns.heatmap(p2p_cov, cmap='RdYlBu_r', ax=ax, center=0, vmin=-1,vmax=1)
ax.set_title('Inter-percept RSM, raw')
ax.set_xlabel('Percept id')
ax.set_ylabel('Percept id')

# comptue the projected precepts
proj_percepts = [np.zeros((T, n_dim)) for j in range(bank_size)]
for j in range(bank_size):
    for t in range(T):
        proj_percepts[j][t, :] = rand_proj[j] @ percepts[t]

# compute/plot the RSM for the projected percepts
for j in range(bank_size):
    pp2p_cov = np.corrcoef(proj_percepts[j])

    f, ax = plt.subplots(1,1, figsize=(5,4))
    sns.heatmap(pp2p_cov, cmap='RdYlBu_r', ax=ax, center=0, vmin=-1,vmax=1)
    ax.set_title(f'Inter-percept RSM, projected {j}')
    ax.set_xlabel('Percept id')
    ax.set_ylabel('Percept id')


# compute the RSM for the same percept projected by different projection mats
# i.e. does different random projection create different looks?
proj_contexts = [np.zeros((bank_size, n_dim)) for j in range(n_context)]
for j in range(n_context):
    for b in range(bank_size):
        proj_contexts[j][b] = rand_proj[b] @ contexts[j]


# compute/plot the RSM for the projected context
for j in range(n_context):
    pp2p_cov = np.corrcoef(proj_contexts[j])
    f, ax = plt.subplots(1,1, figsize=(5,4))
    sns.heatmap(pp2p_cov, cmap='RdYlBu_r', ax=ax, center=0, vmin=-1,vmax=1)
    ax.set_title(f'RSM for projected context, context {j}')
    ax.set_xlabel('Projection id')
    ax.set_ylabel('Projection id')

'''conclusion
the same vector projected by different random vector have different "looks"
after random projection, relatively distrances (RSM) is preserved
this means the idea of having a bank of CRPs might work
'''
