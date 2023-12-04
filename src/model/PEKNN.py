import numpy as np
from utils import random_ortho_mat

class PEKNN():
    '''a prediction error based 1NN
    if PE is high, it make a decision between
        - assigns new input to the cloest exisiting cluster
        - or create a new cluster
    else, it assigns new input to the previous cluster
    '''
    def __init__(
        self, dist_threshold, pe_threshold, k=10, context_dim=32, noise=0
        ):
        """init the PE-KNN

        Parameters
        ----------
        dist_threshold : float
            the distance threshold for deciding whether the new input is far
            away from all existing clusters, in which case a new cluster will be created
        pe_threshold : float
            the PE threshold for deciding whether to activate the PE-KNN,
            otherwise it will assign the input to the previous cluster
        k: int
            the number of minimal distance points in a given cluster used ...
            to compute the distance between a new point to that given cluster

        """
        self.pe_threshold = pe_threshold
        self.dist_threshold = dist_threshold
        self.k = k
        self.context_dim = context_dim
        self.context = []
        self.noise=noise
        # initialization variable
        self.cluster_inited = False
        self.ortho_mat = random_ortho_mat(self.context_dim)

    def update_pe_threshold(self, new_pe_threshold):
        self.pe_threshold = new_pe_threshold

    def add_new_context(self, loc=0, scale=1):
        '''
        sample a random vector to represent the k-th context
        this is useful because
        - random vectors are easy to get
        - random vectors are roughly orthogonal
        '''
        self.context.append(
            np.random.normal(loc=loc, scale=scale, size=(self.context_dim,))
        )
        # self.context.append(
        #     self.ortho_mat[len(self.context)]
        # )
    def get_cluster_counts(self, cluster_id):
        return np.sum(self.cluster_assignment == cluster_id)

    def get_lr_weight(self, cluster_id=None, pseudo_count = 10, denom = 1000):
        if cluster_id is None:
            return 1
        assert 0 < pseudo_count < denom
        cluster_count = self.get_cluster_counts(cluster_id)
        learning_rate_weight = cluster_count + pseudo_count / denom
        return np.clip(learning_rate_weight, a_min=None, a_max=1)


    def init_cluster_assignment(self, x_t):
        """init the cluster assignment, call this at time 0

        Parameters
        ----------
        x_t : 1d array
            input vector, the observation of PE-KNN
        """
        if self.cluster_inited:
            raise ValueError('cluster assignment has been initialized')
        self.cluster_assignment = [0]
        self.x_history = [x_t]
        self.add_new_context()
        self.cluster_inited = True
        return self.cluster_assignment[-1], self.context[self.cluster_assignment[-1]]

    def forward(self, x_t, pe_t=None):
        # if pe is very high or if PE is undefined ...
        if pe_t is None or pe_t > self.pe_threshold:
            # perform the general cluster assignment process
            self.assign_to_cluster(x_t)
        else:
            # if low PE, use the previous cluster
            self.assign_to_prev_cluster(x_t)
        return self.cluster_assignment[-1], self.context[self.cluster_assignment[-1]]

    def assign_to_prev_cluster(self, x_t):
        """assign x_t to the previous cluster

        Parameters
        ----------
        x_t : 1d array
            input vector, the observation of PE-KNN
        """
        self.cluster_assignment.append(self.cluster_assignment[-1])
        self.x_history.append(x_t)
        return self.cluster_assignment[-1], self.context[self.cluster_assignment[-1]]

    def assign_to_cluster(self, x_t):
        mindist2clusters = self.compute_dists2clusters(x_t)
        # decide whether to add a new cluster
        self.split_or_assign(mindist2clusters)
        return self.cluster_assignment[-1], self.context[self.cluster_assignment[-1]]

    def compute_dists2clusters(self, x_t):
        '''compute the minimal distance of x_t to different clusters
        x_history is x_0, x_1, ..., x_t-1
        self.cluster_assignment is the cluster assignment of x_history
        '''
        # observations from different clusters
        existing_clusters_t = np.unique(self.cluster_assignment)
        # loop over all clusters
        mindist2clusters = np.zeros(len(existing_clusters_t))
        for ci, c in enumerate(existing_clusters_t):
            # find all previous observations from this cluster
            mask = np.array(self.cluster_assignment) == c
            all_obs_in_c = np.array(self.x_history)[mask]
            # find the closest distance to this cluster
            all_dists_c = np.sqrt(np.sum((all_obs_in_c - x_t)**2, axis=1))
            # use k minimal distance pts to the dist to this cluster
            min_dist_pts_ids = np.argsort(all_dists_c)[:self.k]
            mindist2clusters[ci] = np.min([all_dists_c[i] for i in min_dist_pts_ids])

        self.x_history.append(x_t)
        return mindist2clusters

    def split_or_assign(self, mindist2clusters):
        # add - if all clusters are far away
        if np.all(mindist2clusters > self.dist_threshold):
            # assign to new cluster
            # if there are n clusters, the new cluster id is n
            num_exisiting_clusters = len(np.unique(self.cluster_assignment))
            self.cluster_assignment.append(num_exisiting_clusters)
            self.add_new_context()
        else:
            # else, assign to the closest cluster
            self.cluster_assignment.append(np.argmin(mindist2clusters))




if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from task import CSW
    from utils import compute_smth_mean
    from scipy.spatial import distance_matrix
    sns.set(style='white', palette='colorblind', context='poster')
    np.random.seed(0)

    '''simulate the PE'''
    # event_len = 8
    # n_events = 32
    # block_size = 16
    event_len = 10
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
    smth_X = np.array([compute_smth_mean(x_i, w_cur) for x_i in X])
    observations = smth_X[:,-1,:]
    np.shape(smth_X)

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
    dist_threshold = .1
    pe_threshold = 8
    peknn = PEKNN(dist_threshold, pe_threshold)

    for t in range(T):
        # assign to 0 for the 1st obs
        if t == 0:
            peknn.init_cluster_assignment(nsy_obs[t])
            continue
        # if low PE, use the previous cluster
        if pe[t] < peknn.pe_threshold:
            peknn.assign_to_prev_cluster(nsy_obs[t])
            continue

        # perform the general cluster assignment process

        peknn.assign_to_cluster(nsy_obs[t])
        print(np.shape(peknn.cluster_assignment))


    f, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(pe, label='PE')
    axes[0].axhline(pe_threshold, label='threshold', color='red', linestyle='--')
    axes[0].set_ylabel('PE')

    axes[1].plot(ctx_ids, label='ground truth context', alpha = .5)
    axes[1].plot(peknn.cluster_assignment, label='inferred context', alpha = .5)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('context id')
    axes[1].legend()
    sns.despine()
