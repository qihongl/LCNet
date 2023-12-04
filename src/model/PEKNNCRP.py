import numpy as np


class PEKNNCRP():
    '''a prediction error based 1NN
    if PE is high, it make a decision between
        - assigns new input to the cloest exisiting cluster
        - or create a new cluster
    else, it assigns new input to the previous cluster

    use PEKNN to compute likelihood and combine with CRP prior to get posterior
    '''
    def __init__(self, dist_threshold, pe_threshold, alpha=16, p_threshold=.15,context_dim=32):
        """init the PE-KNN

        Parameters
        ----------
        dist_threshold : float
            the distance threshold for deciding whether the new input is far
            away from all existing clusters, in which case a new cluster will be created
        pe_threshold : float
            the PE threshold for deciding whether to activate the PE-KNN,
            otherwise it will assign the input to the previous cluster

        """
        assert alpha > 0 and pe_threshold > 0
        self.alpha = alpha
        self.pe_threshold = pe_threshold
        self.p_threshold = p_threshold
        self.dist_threshold = dist_threshold
        self.context_dim = context_dim
        self.context = []
        self.cluster_inited = False
        self.counts = []

    def increase_count(self, context_id):
        self.counts[context_id] +=1

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
        self.counts.append(0)


    def forward(self, x_t, pe_t=None):
        # if pe is very high or if PE is undefined ...
        if pe_t is None or pe_t > self.pe_threshold:
            # perform the general cluster assignment process
            self.assign_to_cluster(x_t)
        else:
            # if low PE, use the previous cluster
            self.assign_to_prev_cluster(x_t)
        return self.cluster_assignment[-1], self.context[self.cluster_assignment[-1]]

    def init_cluster_assignment(self, x_t):
        """init the cluster assignment, call this at time 0

        Parameters
        ----------
        x_t : 1d array
            input vector, the observation of PE-KNN
        """
        self.cluster_inited = True
        self.cluster_assignment = [0]
        self.x_history = [x_t]
        self.add_new_context()
        self.counts[0] += 1
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
        self.counts[self.cluster_assignment[-1]] += 1
        return self.cluster_assignment[-1], self.context[self.cluster_assignment[-1]]

    def assign_to_cluster(self, x_t):
        mindist2clusters = self.compute_dists2clusters(x_t)
        likelihood = to_probs(mindist2clusters)
        prior = np.array([self.counts[k] / np.sum(self.counts)
                    for k in range(len(self.context))])
        posterior = likelihood * prior
        # add p for new class
        posterior = np.append(posterior, self.alpha / np.sum(self.counts))
        # normalization
        posterior /= posterior.sum()

        # print(mindist2clusters)
        print('lik')
        print(likelihood)
        print('prior')
        print(prior, self.counts)
        print('pos')
        print(posterior)
        print()

        # decide whether to add a new cluster
        self.split_or_assign(posterior)
        self.counts[self.cluster_assignment[-1]] += 1

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
            # use the minimal distance for this cluster, correspond to 1NN
            mindist2clusters[ci] = np.min(all_dists_c)
        self.x_history.append(x_t)
        return mindist2clusters

    def split_or_assign(self, posterior):
        # add a new cluster
        if posterior[-1] > self.p_threshold:
            num_exisiting_clusters = len(np.unique(self.cluster_assignment))
            # if there are n clusters, the new cluster id is n
            self.cluster_assignment.append(num_exisiting_clusters)
            self.add_new_context()
        else:
            self.cluster_assignment.append(np.argmax(posterior[:-1]))


    # def split_or_assign(self, mindist2clusters):
    #     # add - if all clusters are far away
    #     if np.all(mindist2clusters > self.dist_threshold):
    #         num_exisiting_clusters = len(np.unique(self.cluster_assignment))
    #         # if there are n clusters, the new cluster id is n
    #         self.cluster_assignment.append(num_exisiting_clusters)
    #         self.add_new_context()
    #     else:
    #         # else, assign to the closest cluster
    #         self.cluster_assignment.append(np.argmin(mindist2clusters))

    # def compute_crp_prior(self):
    #     prior = np.array([self.counts[k] / np.sum(self.counts) for k in range(len(self.context))])
    #     # prior.append(self.alpha / np.sum(self.counts))
    #     return np.array(prior)


def to_probs(dists):
    '''
    convert to a probability distribution
    '''
    # make sure it is a 1d array
    assert dists.ndim == 1
    return dists / dists.sum()


if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from task import CSW
    from utils import compute_smth_mean
    from scipy.spatial import distance_matrix
    sns.set(style='white', palette='colorblind', context='poster')


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
    smth_X = np.array([compute_smth_mean(x_i, w_cur) for x_i in X])
    observations = smth_X[:,-1,:]

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
    pe_threshold = 3
    pkcrp = PEKNNCRP(dist_threshold, pe_threshold)

    for t in range(T):
        # assign to 0 for the 1st obs
        if t == 0:
            pkcrp.init_cluster_assignment(nsy_obs[t])
            continue
        # if low PE, use the previous cluster
        if pe[t] < pkcrp.pe_threshold:
            pkcrp.assign_to_prev_cluster(nsy_obs[t])
            continue

        # perform the general cluster assignment process
        pkcrp.assign_to_cluster(nsy_obs[t])
        # print(np.shape(pkcrp.cluster_assignment))



    f, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(pe, label='PE')
    axes[0].axhline(pe_threshold, label='threshold', color='red', linestyle='--')
    axes[0].set_ylabel('PE')

    axes[1].plot(ctx_ids, label='ground truth context')
    axes[1].plot(pkcrp.cluster_assignment, label='inferred context')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('context id')
    axes[1].legend()
    sns.despine()
