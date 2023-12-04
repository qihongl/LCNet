import numpy as np
from utils import random_ortho_mat



class SimpleContext():

    def __init__(self, context_dim, concentration, loss_q_size=10, pseudo_count=10, stickiness=10):
        self.context_dim = context_dim
        self.loss_q_size = loss_q_size
        self.pseudo_count = pseudo_count
        self.stickiness = stickiness
        self.concentration = concentration
        self.reset_context()
        self.reset_loss_queue()

    def reset_context(self):
        self.n_context = 0
        self.prev_cluster_id = None

        # self.ortho_mat = random_ortho_mat(self.context_dim)
        # self.context = [self.ortho_mat[0]]
        # self.context = [np.mean(self.ortho_mat,axis=0)]
        self.context = [self.random_vector()]
        # self.context = [self.zero_vector()]
        # self.counts = [self.pseudo_count]
        self.counts = [self.concentration]

    def reset_loss_queue(self):
        self.loss_queue = []

    def add_new_context(self):
        '''
        sample a random vector to represent the k-th context
        this is useful because
        - random vectors are easy to get
        - random vectors are roughly orthogonal
        '''
        new_context = self.random_vector()
        # new_context = self.ortho_mat[self.n_context+1]
        self.context.append(new_context)
        self.counts.append(self.pseudo_count)
        self.n_context += 1
        # self.prev_cluster_id = 1
        return self.n_context, new_context

    def compute_posterior(self, likelihood, verbose=False):
        stickiness_vec = np.zeros(self.n_context+1)
        stickiness_vec[self.prev_cluster_id] = self.stickiness
        # compute the sticky CRP prior
        z = np.sum(self.counts) + self.stickiness
        prior = (np.array(self.counts) + stickiness_vec) / z
        # prior = np.array([
        #     self.counts[k] + stickiness_vec[k] / z for k in range(self.n_context+1)
        # ])
        if verbose:
            print('counts (1st one is the new context) = ', np.array(self.counts) + stickiness_vec)
            print('z = ', z)
            print('prior = ', prior)
        # posterior = likelihood * prior
        # posterior /= posterior.sum()
        # return posterior
        return likelihood * prior


    def assign_context(self, posterior, get_context_vector=False, verbose=True):
        ctx_id = np.argmax(posterior)
        # n_context_out_of_bound = len(self.context) >= self.context_dim
        # get the context vector
        if ctx_id == 0:
            ctx_id, ctx_vec = self.add_new_context()
            if verbose:
                print(f'posterior: {posterior} \t adding the {self.n_context}-th context!')
        else:
            # if n_context_out_of_bound:
            #     ctx_id = np.argsort(posterior)[-2]
            #     print('WARNING: reached max number of contexts!')
            ctx_vec = self.context[ctx_id]
            if verbose:
                print(f'posterior: {posterior} \t reusing {ctx_id}-th context!')
        self.counts[ctx_id] += .01
        self.prev_cluster_id = ctx_id
        if get_context_vector:
            return ctx_id, ctx_vec
        return ctx_id

# np.argsort(np.array([0,3,1]))

    def record_loss_t(self, loss_t):
        self.loss_queue.append(loss_t)
        # update the record of recent errors
        if len(self.loss_queue) > self.loss_q_size:
            self.loss_queue.pop(0)

    def random_vector(self, loc=0, scale=1):
        return np.random.normal(loc=loc, scale=scale, size=(self.context_dim,))

    def zero_vector(self):
        return np.zeros(self.context_dim,)




if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    sns.set(style='white', palette='colorblind', context='poster')

    context_dim=32
    loss_q_size=10
    pseudo_count=1
    sc = SimpleContext(context_dim=context_dim, loss_q_size=loss_q_size, pseudo_count=pseudo_count)

    sc.counts
    c_it, ctx_it = sc.add_new_context()

    likelihood = [.5, .5]
    posterior = sc.compute_posterior(likelihood)
    print(posterior)
    np.argmax(posterior)
    print(sc.counts)
    print(sc.n_context)


    # c_it, ctx_it = sc.assign_context(posterior, verbose=True)
