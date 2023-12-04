import numpy as np
from utils import to_pth

# constants
DIM_CONTEXT = 2 # the dimension fo the context vector
B = 2
CTX_VAL_2_LR = {0 : 'l', 1: 'r'}


class MixedCSW():
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

    the context id can be [0, 0], [0, 1], [1, 0], [1, 1], where
    [0, x] -> L on all even levels
    [1, x] -> R on all even levels
    [x, 0] -> L on all odd  levels
    [x, 1] -> R on all odd  levels
    '''

    def __init__(self, event_len=4, vec_dim=None, ctx_dim=16, noise=0, reptype='onehot'):
        assert event_len % 2 == 0, 'event_len must be even'
        self.event_len = event_len
        self.vec_dim = vec_dim
        self.ctx_dim = ctx_dim
        self.noise = noise
        self.reptype = reptype
        self._gen_all_nodes()
        self._gen_ctx_rep()

    def _gen_ctx_rep(self):
        self.ctx_evn = [np.random.uniform(size=self.ctx_dim) for _ in range(2)]
        self.ctx_odd = [np.random.uniform(size=self.ctx_dim) for _ in range(2)]

    def get_ctx_rep(self, context_ids):
        i, j = context_ids
        assert i in [0, 1] and j in [0, 1], 'context id must be 0 or 1'
        # context_rep = np.concatenate([self.ctx_evn[i], self.ctx_odd[j]])
        # if to_pytorch:
        #     context_rep = to_pth(context_rep)
        # return context_rep
        return self.ctx_evn[i], self.ctx_odd[j]

    def _gen_all_nodes(self):
        if self.reptype == 'onehot':
            assert self.vec_dim is None, \
                f'vector dim is specified to be {self.vec_dim} but it will not be used for one hot'
            # number of nodes need to = event length x branching factor of the graph
            self.n_nodes = self.event_len * B
            # vec dim = n nodes, due to the identity node matrix
            self.vec_dim = self.n_nodes
            # identity node matrix
            self.node_reps = np.eye(self.n_nodes)
        elif self.reptype == 'normal':
            assert self.vec_dim is not None, 'vector dim cannot be none'
            # number of nodes need to = event length x branching factor of the graph
            self.n_nodes = self.event_len * B
            # identity node matrix
            self.node_reps = np.random.normal(size=(self.n_nodes, self.vec_dim))
        else:
            raise NotImplementedError('unrecognized representation type')

    def sample(self, n_data, to_pytorch=False):
        '''sample paths from random context ids'''
        paths = np.empty((n_data, self.event_len, self.vec_dim))
        context_ids = np.empty((n_data, 2))
        # generate n paths
        for i in range(n_data):
            paths[i], context_ids[i] = self.sample_path()
        # to pytorch format
        if to_pytorch:
            paths = to_pth(paths)
        return paths, context_ids

    def sample_compositional(self, n_train, to_pytorch=False):
        """generate samples for a compositional curriculum where
        during the training phase, the model only see context 00 and 11
        during the test     phase, the model only see context 01 and 10
        so the model has to make predictions on novel context combination

        Parameters
        ----------
        n_train : int
            the number of training trials,
            the total number of trials will be n_train x 2
        to_pytorch : bool
            whether to convert to pytorch data

        Returns
        -------
        array (n_data x event_length x vector_dim), array (n_data x 2)
            Description of returned object.

        """
        assert n_train % 2 == 0, 'number of training data need to be even'
        context_ids = self.get_blocked_context_ids(n_train//2)
        n_data = len(context_ids)
        paths = np.empty((n_data, self.event_len, self.vec_dim))
        # generate n paths
        for i in range(n_data):
            paths[i] = self.make_path(context_ids[i])
        # to pytorch format
        if to_pytorch:
            paths = to_pth(paths)
        return paths, context_ids

    def sample_path(self):
        context_ids = self.sample_context_ids()
        path = self.make_path(context_ids)
        return path, context_ids

    def sample_context_ids(self):
        '''sample the values for the context vector
        can be [0, 0], [0, 1], [1, 0], [1, 1]
        '''
        return [int(np.round(np.random.uniform())) for i in range(DIM_CONTEXT)]

    def make_path(self, context_ids):
        node_ids_evn = self.get_node_ids(context_ids[0], True)
        node_ids_odd = self.get_node_ids(context_ids[1], False)
        path = np.empty((self.event_len, self.vec_dim))
        path[0::2] = self.node_reps[node_ids_evn]
        path[1::2] = self.node_reps[node_ids_odd]
        if self.noise > 0:
            path += np.random.normal(loc=0,scale=self.noise, size=np.shape(path))
        return path

    def get_node_ids(self, ctx_id, even):
        '''assume
        1st half of the nodes are L nodes
        2nd half of the nodes are R nodes
        '''
        assert ctx_id in [0, 1]
        lr = CTX_VAL_2_LR[ctx_id]
        # choose l vs. r nodes
        if lr.lower() == 'l':
            node_ids = np.arange(self.n_nodes//2)
        else:
            node_ids = np.arange(self.n_nodes//2, self.n_nodes)
        # select even vs odd ids
        if even:
            return node_ids[0::2]
        return node_ids[1::2]

    def get_blocked_context_ids(self, block_size, testorder=False):
        '''define the context ids sequence for the following curriculum:
        during the training phase,
        - the model only sees context [0,0] or [1,1] in a blocked manner -> i.e., path 0123 or 4567
        '''
        blocked_context_ids = []
        # choose schema order
        if testorder:
            ordering = [[0,0], [1,1], [0,1], [1,0]]
        else:
            ordering = self.random_schema_order()
        # generate context id
        for i, j in ordering:
            for _ in range(block_size):
                blocked_context_ids.append([i, j])
        return np.array(blocked_context_ids)

    def random_schema_order(self):
        ordering_p1 = [[0,0], [1,1]]
        ordering_p2 = [[0,1], [1,0]]
        # shuffle order
        np.random.shuffle(ordering_p1)
        np.random.shuffle(ordering_p2)
        # combine p1 and p2
        ordering = ordering_p1 + ordering_p2
        # randomly reverse p1 p2 order
        if np.random.uniform() > .5:
            return ordering[::-1]
        return ordering

    # def get_randomized_blocked_context_ids(self, block_size):
    #     '''define the context ids sequence for the following curriculum:
    #     during the training phase,
    #     - the model only sees context [0,0] or [1,1] in a RANDOM ORDER-> i.e., path 0123 or 4567
    #     '''
    #     blocked_context_ids = []
    #     train_context_ids = [[0,0], [1,1]]
    #     test_context_ids = [[0,1], [1,0]]
    #     odering = np.random.permutation([0] * block_size + [1] * block_size)
    #     for o in odering:
    #         blocked_context_ids.append(train_context_ids[o])
    #     odering = np.random.permutation([0] * block_size + [1] * block_size)
    #     for o in odering:
    #         blocked_context_ids.append(test_context_ids[o])
    #     return np.array(blocked_context_ids)


    def decode_node_id(self, node_rep):
        node_id = np.where(np.all(node_rep == task.node_reps, axis=1))[0]
        assert len(node_id) == 1, 'input should correspond to a unique node id'
        return int(node_id)

    def decode_node_ids(self, path):
        return [self.decode_node_id(n) for n in path]


if __name__ == "__main__":
    '''how to use'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools
    sns.set(style='white', palette='colorblind', context='poster')
    event_len = 4
    noise = 0
    vec_dim, reptype = None, 'onehot'
    # vec_dim, reptype = 13, 'normal'
    task = MixedCSW(event_len=event_len, vec_dim=vec_dim, reptype=reptype, noise=noise)

    block_size = 1
    for i, j in task.get_blocked_context_ids(block_size):
        path = task.make_path([i, j])
        print(f'context id: {[i, j]} | path: {task.decode_node_ids(path)}' )

    n_train = 64
    paths, context_ids = task.sample_compositional(n_train, to_pytorch=False)
    plt.plot(np.array(context_ids))


    f, ax = plt.subplots(1,1, figsize=(10,5))
    ax.plot(np.array(context_ids)[:,0], label='latent cause A')
    ax.plot(np.array(context_ids)[:,1], label='latent cause B')
    ax.axvline(n_train, linestyle = '--', color='red', label ='start testing')
    ax.set_ylabel('True latent cause')
    ax.set_xlabel('Trials')
    ax.legend()
    sns.despine()
