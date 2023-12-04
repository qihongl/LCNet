import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from task import CSW
from sklearn.decomposition import PCA
from utils import compute_smth_mean, to_sqnp
sns.set(style='white', palette='colorblind', context='poster')
cp2 = sns.color_palette(n_colors=2)

# setup
event_len = 4
noise = 0
reptype = 'onehot'
# reptype = 'onehot'
csw = CSW(event_len=event_len, noise=noise, reptype=reptype)

# take a sample
n_sample = 8
block_size = 4
ctx_ids, np_seqs, node_seqs, X = csw.sample(n_sample, block_size)



# compute the smoothed average
w_cur = 1
smth_X = np.array([compute_smth_mean(x_i, w_cur) for x_i in to_sqnp(X)])
#
# smth_X[smth_X > 0] = 1
# smth_X[smth_X <= 0] = 0


# show one trial and its smoothed x
i = 0
f, ax = plt.subplots(1,1, figsize=(6,5))
ax.imshow(smth_X[i], cmap='bone')
ax.set_title('smoothed')
f, ax = plt.subplots(1,1, figsize=(6,5))
ax.imshow(X[i], cmap='bone')
ax.set_title('X')


pca = PCA(n_components=2)

t = 3
smth_X
pcsmth_X = pca.fit_transform(smth_X[:, t, :])

np.shape(smth_X)

f, ax = plt.subplots(1,1, figsize=(8,6))
for ctx_i, color_i in zip(np.unique(ctx_ids), cp2):
    ax.scatter(
        x=pcsmth_X[ctx_ids==ctx_i,0], y=pcsmth_X[ctx_ids==ctx_i,1],
        color=color_i, label=f'context {ctx_i}'
    )
ax.set_title('Smoothed input sequences')
ax.set_xlabel('pc 1')
ax.set_ylabel('pc 2')
ax.legend()
sns.despine()


'''summed input seqs doesn't work as well '''
# sum_X = np.array([np.sum(x_i,axis=0) for x_i in X])
#
# f, ax = plt.subplots(1,1, figsize=(8,6))
# for ctx_i, color_i in zip(np.unique(ctx_ids), cp2):
#     ax.scatter(
#         x=sum_X[ctx_ids==ctx_i,0], y=pcsmth_X[ctx_ids==ctx_i,1],
#         color=color_i, label=f'context {ctx_i}'
#     )
# ax.set_title('Summed input sequences')
# ax.set_xlabel('pc 1')
# ax.set_ylabel('pc 2')
# ax.legend()
# sns.despine()

# plt.imshow(, aspect='auto')
cb = sns.color_palette('Blues', n_colors = event_len)
co = sns.color_palette('Oranges', n_colors = event_len)

pca_fitall = pca.fit(np.reshape(smth_X, (-1, csw.vec_dim)))
pcsmth_X = np.array([pca_fitall.transform(smth_X[:,t,:]) for t in range(event_len)])
pcsmth_X += np.random.normal(scale=.01, size=np.shape(pcsmth_X))

f, ax = plt.subplots(1,1, figsize=(8,6))
noise_level = .005
noise = np.random.normal(scale=noise_level, size=(2, 2))

n = np.shape(pcsmth_X)[1]
for i in range(n):
    if ctx_ids[i] == 0:
        color_i = co
        noise_i = noise[0]
    else:
        color_i = cb
        noise_i = noise[1]

    traj_i = pcsmth_X[:,i,:] + noise_i

    ax.plot(traj_i[:,0], traj_i[:,1], linestyle='-', color='grey', alpha=.3)
    for j, pts in enumerate(traj_i):
        ax.scatter(pts[0], pts[1], color=color_i[j], marker='o', zorder=999)

ax.set_title('Smoothed input sequences')
ax.set_xlabel('pc 1')
ax.set_ylabel('pc 2')
# ax.legend()
sns.despine()
