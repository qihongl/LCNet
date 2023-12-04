import numpy as np
from copy import deepcopy
from utils import to_np, to_pth, sample_binary_array, flip_bits
# from utils import ortho_proj

# constants
N_CONTEXTS = 2
B = 2
N_POSSIBLE_PATHS = 2  # possible paths per context
CONTEXT_IDS = [0, 1]


class CSW():
    '''the T x B coffee shop world, where B = 2

    graph:
    here's the CSW with event_len = 3 (see andre/silvy's paper):
    1 - 3 - 5 - 7
      x   x   x
    2 - 4 - 6 - 8

    but let's use 0-based indexing:
    0 - 2 - 4 - 6
      x   x   x
    1 - 3 - 5 - 7

    task definition / stimuli representation:
    at the beginning (t=0) of a trial, csw outputs a percept (a binary vector)
    at time t = 0, 1, 2, csw outputs a node (one hot)
    the task is to predict the upcoming node given the percept and node at t

    schemas:
    context 1 uses '-' edges only and context 2 uses 'x' only
    - context 1 samples: 0 2 4 6; 1 3 5 7
    - context 2 samples: 0 3 4 7; 1 2 5 6
    the formula to get all possible samples
    - context 1:
        - row 1 (== all even nums)
        - row 2 (== all odd nums)
    - context 2:
        - interleave even terms from row 1 & odd terms from row 2
        - interleave odd terms from row 1 & even terms from row 2

    transition structure:
    at the beginning, the two nodes are equally likely,
    then the rest of the edges ('x' and '-') has probability 1
    '''

    def __init__(self, event_len=5, n_blocks_per_ctx=2, noise=0, reptype='onehot', test_mode=True):
        self.event_len = event_len
        self.n_blocks_per_ctx = n_blocks_per_ctx
        self.n_blocks_total = n_blocks_per_ctx * N_CONTEXTS
        self.noise = noise
        self.reptype = reptype
        self.test_mode = test_mode
        self._gen_all_nodes()
        #
        self.reset(test_mode=test_mode, verbose=False)


    def reset(self, test_mode, verbose=True):
        '''reset the perceptual inputs for each context'''
        assert type(test_mode) == type(True)
        self.test_mode = test_mode
        self._gen_percepts()
        self._gen_all_paths()
        if verbose:
            print(f'test mode is set to {test_mode}')

    def _gen_all_nodes(self):
        if self.reptype == 'onehot':
            # number of nodes need to = event length x branching factor of the graph
            self.n_nodes = self.event_len * B
            # vec dim = n nodes, due to the identity node matrix
            self.vec_dim = self.n_nodes
            # identity node matrix
            self.node_reps = np.eye(self.n_nodes)

        elif self.reptype == 'normal':
            # number of nodes need to = event length x branching factor of the graph
            self.n_nodes = self.event_len * B
            # vec dim = n nodes, due to the identity node matrix
            self.vec_dim = self.n_nodes * 3
            # identity node matrix
            self.node_reps = np.random.normal(size=(self.n_nodes, self.vec_dim))
            # self.node_reps = np.random.uniform(size=(self.n_nodes, self.vec_dim))
        else:
            raise NotImplementedError('')
        # get a dense common start node, zero vector prevents learning
        # self.common_start_node = np.random.normal(size=(self.vec_dim + N_CONTEXTS))
        self.common_start_node = np.random.normal(size=(self.vec_dim))


    def _gen_percepts(self, method='anticorrelated'):
        if method == 'anticorrelated':
            p1 = sample_binary_array(self.vec_dim, 1)
            p2 = 1 - p1
            self.percepts = np.squeeze([p1, p2])
        else:
            self.percepts = sample_binary_array(self.vec_dim, N_CONTEXTS)

    def _gen_two_random_paths(self):
        def _get_random_node(node1, node2):
            if np.random.uniform() > .5:
                return node1
            return node2

        assert B == 2 # this func doesn't work if b is not 2
        all_nodes = np.arange(self.event_len * B)
        path1 = np.zeros(self.event_len)
        path2 = np.zeros(self.event_len)
        for ii, i in enumerate([2*n for n in range(self.event_len)]):
            node_l = i
            node_r = i+1
            # the "balanced" scheme
            # if np.random.uniform() > .5:
            #     path1[ii], path2[ii] = node_l, node_r
            # else:
            #     path1[ii], path2[ii] = node_r, node_l
            path1[ii] = _get_random_node(node_l, node_r)
            path2[ii] = _get_random_node(node_l, node_r)
        return path1, path2


    def _gen_all_paths(self):
        self.paths = [
            np.zeros((N_POSSIBLE_PATHS, self.event_len), dtype=np.int)
            for _ in range(N_CONTEXTS)
        ]
        if self.test_mode:
            row1 = np.arange(self.event_len * B)[0::2]
            row2 = np.arange(self.event_len * B)[1::2]
            # context 0
            # path 0
            self.paths[0][0] = row1
            # path 1
            self.paths[0][1] = row2
            # context 1
            # path 0
            self.paths[1][0][0::2] = row1[0::2]
            self.paths[1][0][1::2] = row2[1::2]
            # path 1
            self.paths[1][1][1::2] = row1[1::2]
            self.paths[1][1][0::2] = row2[0::2]
        else:
            # print('on')
            self.paths[0][0], self.paths[0][1] = self._gen_two_random_paths()
            self.paths[1][0], self.paths[1][1] = self._gen_two_random_paths()


    def sample(self, block_size=16, to_torch=False, condition='blocked'):
        """sample some number of trials

        Parameters
        ----------
        block_size : int
            the size (# trials) of the block
        to_torch : bool
            whether convert to pytorch format

        Returns
        -------
        ctx_ids : list of int
            the context id at time t
        np_seqs : n_total_trials x perpcet_dim
            the sequence of noisy percept
        node_seqs: n_total_trials x event_len
            the sequence of nodes
        vec_seqs : n_total_trials x event_len x vec_dim/x_dim


        """
        # block_range = [2**k for k in range(1 + int(np.log2(n_sample)))]
        # assert block_size in block_range, \
        #     f'block_size == {block_size}, must be in {block_range}'
        # get context id with a particular block size
        # ctx_ids = get_blocked_sequence(block_size, n_sample)

        # seq1 = get_blocked_sequence(block_size, block_size)
        # seq2 = get_blocked_sequence(1, block_size)
        # ctx_ids = np.concatenate([seq1] * 1 + [seq2] * 3)
        # # ctx_ids = np.concatenate([seq2] * 1 + [seq1] * 1 + [seq2] * 2)
        if self.test_mode:
            rand_seq = get_random_seq(block_size, n_ctx=N_CONTEXTS)
            if condition == 'blocked':
                # blocked-1st
                seq1 = get_blocked_sequence(block_size, block_size, n_ctx=N_CONTEXTS)
                ctx_ids = np.concatenate([seq1] * self.n_blocks_per_ctx  + [rand_seq] * 1)
                # ctx_ids = np.concatenate([seq1] * 1 + [seq2] * 1 + [rand_seq] * 2)
            elif condition == 'interleaved':
                # interleaved-1st
                seq2 = get_blocked_sequence(1, block_size, n_ctx=N_CONTEXTS)
                ctx_ids = np.concatenate([seq2] * self.n_blocks_per_ctx + [rand_seq] * 1)
                # ctx_ids = np.concatenate([seq2] * 1 + [seq1] * 1 + [rand_seq] * 2)
            else:
                raise ValueError(f'Unrecognizable condition = {condition}')
        else:
            ctx_ids = np.concatenate(
                [get_random_seq(block_size, n_ctx=N_CONTEXTS)
                for _ in range(self.n_blocks_per_ctx + 1)]
            )

        self.n_total_trials = len(ctx_ids)

        # path_ids = np.tile([0, 1], self.n_total_trials // 2)
        # path_ids = np.tile([0, 0, 1, 1], self.n_total_trials // 4)
        node_seqs = np.zeros((self.n_total_trials, self.event_len))
        vec_seqs = np.zeros((self.n_total_trials, self.event_len, self.vec_dim))
        np_seqs = np.zeros((self.n_total_trials, self.vec_dim), dtype=np.int)
        for i, ctx_id in enumerate(ctx_ids):
            node_seqs[i], vec_seqs[i] = self._sample_1_trial(ctx_id)
            # node_seqs[i], vec_seqs[i] = self._make_1_trial(ctx_id, path_ids[i])
            np_seqs[i] = self.get_noisey_percept(ctx_id)
        # reformat data
        if to_torch:
            vec_seqs = to_pth(vec_seqs)
            # np_seqs = to_pth(np_seqs)
        return ctx_ids, np_seqs, node_seqs, vec_seqs


    # def sample_wctx2(self, block_size=16, to_torch=False, condition=None, add_zero_node=True):
    #     """sample some number of trials
    #     AND use perceptual input as the 1st input vector
    #
    #     Parameters
    #     ----------
    #     n_sample : type
    #         Description of parameter `n_sample`.
    #     block_size : type
    #         Description of parameter `block_size`.
    #     to_torch : type
    #         Description of parameter `to_torch`.
    #
    #     Returns
    #     -------
    #     type
    #         Description of returned object.
    #
    #     """
    #
    #     ctx_ids, np_seqs, node_seqs, vec_seqs = self.sample(
    #         block_size=block_size, to_torch=False, condition=condition)
    #
    #     if add_zero_node:
    #         # every trial start with a zero vector as the common start node
    #         # zero vector
    #         # common_start_node = np.zeros(self.vec_dim + N_CONTEXTS)
    #         final_event_len = self.event_len + 2
    #     else:
    #         final_event_len = self.event_len + 1
    #
    #     # vec_seqs_wctx = np.zeros((self.n_total_trials, final_event_len, self.vec_dim + N_CONTEXTS))
    #     vec_seqs_wctx = np.zeros((self.n_total_trials, final_event_len, self.vec_dim + N_CONTEXTS))
    #
    #     for i in range(self.n_total_trials):
    #         # get context indicative node
    #         ctx_mat_i = np.zeros((self.event_len + 1, 2))
    #         ctx_mat_i[0, ctx_ids[i]] = 1
    #         vec_seqs_i = np.vstack([np.zeros(self.vec_dim,), vec_seqs[i]])
    #         # vec_seqs_wctx[i] = np.hstack([ctx_mat_i, vec_seqs_i])
    #         vec_seqs_wctx_i = np.hstack([ctx_mat_i, vec_seqs_i])
    #         # whether to add the common zero node
    #         if add_zero_node:
    #             vec_seqs_wctx[i] = np.vstack([self.common_start_node, vec_seqs_wctx_i])
    #         else:
    #             vec_seqs_wctx[i] = vec_seqs_wctx_i
    #
    #     if add_zero_node:
    #         vec_seqs_ = np.zeros((self.n_total_trials, self.event_len + 1, self.vec_dim))
    #         for i in range(self.n_total_trials):
    #             vec_seqs_[i] = np.vstack([np.zeros(self.vec_dim), vec_seqs[i]])
    #     else:
    #         vec_seqs_ = vec_seqs
    #
    #     # self.vec_dim = self.vec_dim + N_CONTEXTS
    #     if to_torch:
    #         vec_seqs_wctx = to_pth(vec_seqs_wctx)
    #     return ctx_ids, np_seqs, node_seqs, vec_seqs_, vec_seqs_wctx

    def sample_wctx2(self, block_size=16, to_torch=False, condition=None, add_zero_node=True):
        """sample some number of trials
        AND use perceptual input as the 1st input vector

        Parameters
        ----------
        n_sample : type
            Description of parameter `n_sample`.
        block_size : type
            Description of parameter `block_size`.
        to_torch : type
            Description of parameter `to_torch`.

        Returns
        -------
        type
            Description of returned object.

        """

        ctx_ids, np_seqs, node_seqs, vec_seqs = self.sample(
            block_size=block_size, to_torch=False, condition=condition)

        if add_zero_node:
            # every trial start with a zero vector as the common start node
            # zero vector
            # common_start_node = np.zeros(self.vec_dim + N_CONTEXTS)
            final_event_len = self.event_len + 2
        else:
            final_event_len = self.event_len + 1

        # vec_seqs_wctx = np.zeros((self.n_total_trials, final_event_len, self.vec_dim + N_CONTEXTS))
        vec_seqs_wctx = np.zeros((self.n_total_trials, final_event_len, self.vec_dim))

        for i in range(self.n_total_trials):
            # get context indicative node
            # ctx_mat_i = np.zeros((self.event_len + 1, 2))
            # ctx_mat_i[0, ctx_ids[i]] = 1
            # vec_seqs_i = np.vstack([np.zeros(self.vec_dim,), vec_seqs[i]])

            # vec_seqs_wctx[i] = np.hstack([ctx_mat_i, vec_seqs_i])
            vec_seqs_wctx_i = np.vstack([self.percepts[ctx_ids[i]], vec_seqs[i]])
            # print(vec_seqs_wctx_i)
            # whether to add the common zero node
            if add_zero_node:
                vec_seqs_wctx[i] = np.vstack([self.common_start_node, vec_seqs_wctx_i])
            else:
                vec_seqs_wctx[i] = vec_seqs_wctx_i

        if add_zero_node:
            vec_seqs_ = np.zeros((self.n_total_trials, self.event_len + 1, self.vec_dim))
            for i in range(self.n_total_trials):
                vec_seqs_[i] = np.vstack([np.zeros(self.vec_dim), vec_seqs[i]])
        else:
            vec_seqs_ = vec_seqs

        # self.vec_dim = self.vec_dim + N_CONTEXTS
        if to_torch:
            vec_seqs_wctx = to_pth(vec_seqs_wctx)
        return ctx_ids, np_seqs, node_seqs, vec_seqs_, vec_seqs_wctx


    # def format_y(self, ):
    #     if self.reptype == 'onehot':
    #         y_t = vec_seqs_wctx[i, t + 1]
    #         y_t_1d = csw.to_1D_targ(y_t)
    #     else:
    #         if t == 0:
    #             y_t_1d = ctx_ids[i]
    #             # print(y_t_1d)
    #         else:
    #             y_t_1d = node_seqs[i, t-1]
    #             y_t_1d = is_even(y_t_1d)

    def to_1D_targ(self, y_vector):
        one_hot_position = np.argmax(y_vector)
        # print(one_hot_position, is_even(one_hot_position))
        return is_even(one_hot_position)


    def get_targs(self, context_id, t):
        '''get the two targets at context i, path j, time t
        '''
        i_path0 = self.paths[context_id][0][t]
        i_path1 = self.paths[context_id][1][t]
        return self.node_reps[i_path0], self.node_reps[i_path1]

    # def get_PE(self, context_id, t, y_t, y_hat_t):
    #     '''
    #     compute relative prediction error
    #     between the output and the two possible targets at time t+1
    #     '''
    #     y_t = to_np(y_t)
    #     y_hat_t = to_np(y_hat_t)
    #     targ0, targ1 = self.get_targs(context_id, t + 1)
    #     assert np.all(y_t == targ0) or np.all(y_t == targ1)
    #
    #     y_hat_proj = ortho_proj(y_hat_t, targ0 - targ1)
    #     d1 = np.linalg.norm(y_hat_proj - targ0)
    #     d2 = np.linalg.norm(y_hat_proj - targ1)
    #     d12 = np.linalg.norm(targ0 - targ1)
    #     if d1 > d12 or d2 > d12:
    #         print('omg')
    #     if np.all(y_t == targ0):
    #         return d1, d2
    #     return d2, d1


    def _sample_1_trial(self, context_id):
        assert context_id in CONTEXT_IDS, \
            f'Input Error: context id is {context_id}, must in {CONTEXT_IDS}'
        path_id = int(np.round(np.random.uniform()))
        return self._make_1_trial(context_id, path_id)

    def  _make_1_trial(self, context_id, path_id):
        node_seq = self.paths[context_id][path_id]
        vec_seq = np.array([self.node_reps[i] for i in node_seq])
        return node_seq, vec_seq

    def get_noisey_percept(self, context_id):
        return flip_bits(self.percepts[context_id], self.noise)


def get_blocked_sequence(block_size, n_samples, n_ctx=N_CONTEXTS):
    '''
    form interleaved sequence with block size block_size (=1 -> perfect interleaved)

    seq length = n_samples * n_ctx
    '''
    # block_range = [2**k for k in range(1 + int(np.log2(n_samples)))]
    # assert block_size in block_range, \
    #     f'block_size == {block_size}, must be in {block_range}'

    n_samples_total = n_ctx * block_size
    # get the ids for one block
    ctx_id_one_block = np.reshape(np.repeat(range(n_ctx), block_size), (n_samples_total, 1))
    # compute number of blocks
    n_blocks = n_samples // block_size
    ctx_id_seq = np.reshape(np.vstack([ctx_id_one_block for _ in range(n_blocks)]), (-1))
    return ctx_id_seq


def get_random_seq(n_samples_total, n_ctx=N_CONTEXTS):
    return np.random.choice(range(n_ctx), size=n_samples_total, replace=True)

def is_even(number):
    if number % 2 == 0:
        return 1
    return 0


if __name__ == "__main__":
    '''how to use'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', palette='colorblind', context='poster')

    event_len = 4
    noise = 0
    # reptype = 'normal'
    reptype = 'onehot'
    csw = CSW(event_len=event_len, noise=noise, reptype=reptype)
    csw.reset(test_mode=False)
    f, ax = plt.subplots(1,1, figsize=(10,10))
    node_rep_corr = np.corrcoef(csw.node_reps)
    mask_triu = np.triu(node_rep_corr)
    sns.heatmap(node_rep_corr, mask=mask_triu, square=True, cmap='RdYlBu', ax=ax)
    ax.set_xlabel('node id')
    ax.set_ylabel('node id')

    # f, ax = plt.subplots(1, 1)
    # ax.imshow(csw.percepts)
    # ax.set_title('the two percepts for the two contexts')

    f, ax = plt.subplots(1, 1)
    ax.imshow(csw.node_reps)
    ax.set_title('nodes')

    print('the two possible paths in context 0 \ncsw.paths[0]: \n', csw.paths[0])
    print('the two possible paths in context 1 \ncsw.paths[1]: \n', csw.paths[1])

    # node_seq, vec_seq = csw._sample_1_trial(1)
    #
    block_size = 8
    # ctx_ids, np_seqs, node_seqs, vec_seqs = csw.sample(block_size)
    # np.shape(ctx_ids)
    # np.shape(np_seqs)
    # np.shape(node_seqs)
    # np.shape(vec_seqs)
    # # csw.vec_dim
    #
    # f, ax = plt.subplots(1, 1)
    # ax.plot(ctx_ids)
    # ax.set_title('ctx_ids')
    # ax.set_xlabel('trials')
    # ax.set_ylabel('context id')
    # sns.despine()
    #
    # f, ax = plt.subplots(1, 1)
    # ax.imshow(np_seqs)
    # ax.set_title('the seq of noisy percepts')
    # ax.set_xlabel('percept dim')
    # ax.set_ylabel('trials')
    #
    # for i, ctx_id in enumerate(ctx_ids):
    #     print(i, ctx_id, node_seqs[i, :])
    #
    #
    # # i = 31
    # # f, ax = plt.subplots(1, 1)
    # # ax.imshow(vec_seqs[i], aspect='auto')
    # # ax.set_title(f'the {i}-th trial: \n{node_seqs[i, :]}')
    # # ax.set_xlabel('time')
    # # ax.set_ylabel('the node id rep')
    # # sns.despine()
    # #
    condition = 'blocked'
    # condition = 'interleaved'
    ctx_ids, np_seqs, node_seqs, vec_seqs, vec_seqs_wctx = csw.sample_wctx2(
        block_size, condition=condition)

    # csw.percepts
    np.shape(vec_seqs_wctx)
    i = -4
    f, ax = plt.subplots(1, 1)
    ax.imshow(vec_seqs_wctx[i], aspect='auto')
    ax.set_title(f'the {i}-th trial wctx: \n{node_seqs[i, :]}')
    ax.set_xlabel('time')
    ax.set_ylabel('the node id rep')
    sns.despine()

    f, ax = plt.subplots(1, 1)
    ax.imshow(vec_seqs[i], aspect='auto')
    ax.set_title(f'the {i}-th trial: \n{node_seqs[i, :]}')
    ax.set_xlabel('time')
    ax.set_ylabel('the node id rep')
    sns.despine()

    for i in np.arange(10, 15):
        for v in vec_seqs_wctx[i]:
            one_hot_position = np.argmax(v)
            print(one_hot_position, is_even(one_hot_position))
        print()
