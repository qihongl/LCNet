'''adapet the how-to-use guide from PEKNN, but check if CRP can handle it'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from task import CSW
from model import PECRP
from utils import compute_smth_mean, to_sqnp
from scipy.spatial import distance_matrix
sns.set(style='white', palette='colorblind', context='poster')

for i in range(10):
    '''simulate the PE'''
    # event_len = 8
    # n_events = 32
    # block_size = 16
    event_len = 16
    n_events = 128
    block_size = 64

    T = n_events * 2
    # assume we know the PE threshold
    pe_at_event_boundary = 10

    pe = np.random.uniform(size=T)
    for i in np.arange(block_size, T, block_size):
        # set the PE for the event boundaries
        pe[i] = pe_at_event_boundary

    '''make some events'''
    noise = 0
    reptype = 'onehot'
    # reptype = 'onehot'
    csw = CSW(event_len=event_len, noise=noise, reptype=reptype)
    ctx_ids, np_seqs, node_seqs, X = csw.sample(n_events, block_size)

    # compute smthed mean of the event sequences
    w_cur = .5
    smth_X = np.array([compute_smth_mean(x_i, w_cur) for x_i in to_sqnp(X)])
    # binarize the inputs
    smth_X[smth_X > 0] = 1
    smth_X[smth_X <= 0] = 0

    # take the obs from the last time point
    observations = smth_X[:,-1,:]
    print(np.shape(observations))

    noise = .005
    obs_noise = np.random.normal(scale=noise, size=np.shape(observations))
    nsy_obs = observations + obs_noise

    '''compute the distance matrix of the noisy observations '''
    dist_mat = distance_matrix(nsy_obs, nsy_obs)

    dist_triu = dist_mat[np.triu_indices(np.shape(nsy_obs)[0], k=1)]
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(dist_triu, bins=50)
    ax.set_xlabel('distance')
    ax.set_ylabel('counts')
    sns.despine()


    '''init the clustering module'''
    # dist_threshold = .1
    pe_threshold = 3

    c = .00001
    lmbd = 0
    model = PECRP(c, lmbd=lmbd, context_dim=csw.vec_dim)

    # shuffle(stims)
    # print(f'n context = {len(model.context)}')
    # print("Partition: ", model.partition)
    # print()

    cluster_id = np.zeros(T)
    t, s = 0, observations[0]
    for t, s in enumerate(observations):
        input = np.array(s, dtype=np.int)
        if t == 0:
            cluster_id[t] = model.init_cluster(input)
            continue

        if pe[t] < pe_threshold:
            # assign to prev
            cluster_id[t] = model.assign_to_prev(input)
            continue

        cluster_id[t] = model.stimulate(input)


        # print(f"Stim: {s}\ncluster_id = {cluster_id}")
        # print(f'n context = {len(model.context)}')
        # print("Partition: ", model.partition)
        # print(f'Cluster id = {model.get_cluster_id(s)}')
        # print(f'N = {model.N}')


    f, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(pe, label='PE')
    axes[0].axhline(pe_threshold, label='threshold', color='red', linestyle='--')
    axes[0].set_ylabel('PE')

    axes[1].plot(ctx_ids, label='ground truth context', alpha = .5)
    axes[1].plot(cluster_id, label='inferred context', alpha = .5)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('context id')
    axes[1].legend()
    sns.despine()
