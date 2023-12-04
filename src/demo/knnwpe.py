'''infer the underlying context by using knn with prediction error '''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from task import CSW
from utils import compute_smth_mean, to_sqnp
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
sns.set(style='white', palette='colorblind', context='poster')
cp2 = sns.color_palette(n_colors=2)


sns.set(style='white', palette='colorblind', context='talk')

subj_id = 0
np.random.seed(subj_id)



def compute_dists2clusters(x_t, x_history, cluster_assignment_history):
    '''compute the minimal distance of x_t to different clusters
    x_history is x_0, x_1, ..., x_t-1
    cluster_assignment_history is the cluster assignment of x_history
    '''
    # observations from different clusters
    existing_clusters_t = np.unique(cluster_assignment_history)
    # loop over all clusters
    mindist2clusters = np.zeros(len(existing_clusters_t))
    for ci, c in enumerate(existing_clusters_t):
        # find all previous observations from this cluster
        mask = np.array(cluster_assignment_history) == c
        all_obs_in_c = np.array(x_history)[mask]
        # find the closest distance to this cluster
        all_dists_c = np.sum((all_obs_in_c - x_t)**2, axis=1)
        # use the minimal distance for this cluster, correspond to 1NN
        mindist2clusters[ci] = np.min(all_dists_c)
    return mindist2clusters

def assign(mindist2clusters, cluster_assignment):
    # add - if all clusters are far away
    if np.all(mindist2clusters > dist_threshold):
        cluster_assignment.append(len(np.unique(cluster_assignment)))
    else:
        # else, assign to the closest cluster
        cluster_assignment.append(np.argmin(mindist2clusters))
    return cluster_assignment


'''simulate the PE'''
event_len = 6
n_events = 32
block_size = 16
T = n_events * 2
# assume we know the PE threshold
pe_threshold = 3
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
observations = smth_X[:,-1,:]

noise = .02
obs_noise = np.random.normal(scale=noise, size=np.shape(observations))
nsy_obs = observations + obs_noise

'''visualize the context, the input representation'''
pca = PCA(n_components=2)
pcsmth_X = pca.fit_transform(nsy_obs)

f, ax = plt.subplots(1,1, figsize=(8,6))
for ctx_i, color_i in zip(np.unique(ctx_ids), cp2):
    mask = ctx_ids==ctx_i
    # (nx, ny) = np.random.normal(scale=plt_jitter, size=(2, mask.sum()))
    ax.scatter(
        x=pcsmth_X[mask,0], y=pcsmth_X[mask,1],
        color=color_i, label=f'context {ctx_i}'
    )
ax.set_title('The ground truth context for smoothed input sequences')
ax.set_xlabel('pc 1')
ax.set_ylabel('pc 2')
ax.legend()
sns.despine()

'''compute the distance matrix of the noisy observations '''
dist_mat = distance_matrix(nsy_obs, nsy_obs)

dist_triu = dist_mat[np.triu_indices(np.shape(nsy_obs)[0], k=1)]
f, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(dist_triu, bins=50)
ax.set_xlabel('distance')
ax.set_ylabel('counts')
sns.despine()



'''visualize PE over time'''
f, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(pe)
ax.set_xlabel('t')
ax.set_ylabel('PE')
# ax.legend()
sns.despine()
# plt.plot(ctx_ids)

'''solve the clustering problem'''
k = 1
dist_threshold = .1
# print(np.shape(pe))
# print(np.shape(nsy_obs))



t = 0
for t in range(T):
    # assign
    if t == 0:
        cluster_assignment = [0]
        continue
    # if low PE, use the previous cluster
    if pe[t] < pe_threshold:
        cluster_assignment.append(cluster_assignment[-1])
        continue

    mindist2clusters = compute_dists2clusters(
        nsy_obs[t], nsy_obs[:t], cluster_assignment)
    # decide whether to add a new cluster
    cluster_assignment = assign(mindist2clusters, cluster_assignment)


# plt.plot(cluster_assignment)
