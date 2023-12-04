import numpy as np

def get_anti_corr_vector(dim, vec_set, corr_limit, loop_limit=10000000):
    assert len(vec_set) > 0
    loop_counter = 0
    while True:
        loop_counter +=1
        if loop_counter > loop_limit:
            raise ValueError('loop limit exceeded')
        sample = np.random.normal(loc=0, scale=1, size=(1, dim))
        # sample = np.random.uniform(low=-1, high=1, size=(1, dim))
        corr_with_vec_set = np.corrcoef(vec_set, sample)[-1, :-1]
        if np.all(corr_with_vec_set < corr_limit):
            return np.squeeze(sample)
dim = 96
corr_limit = -.5
n_vecs = 2
# np.shape(np.corrcoef(x1, sample))
# def get_anti_corr_vectors(dim, n_vecs, corr_limit):
x1 = np.random.normal(loc=0, scale=1, size=(dim,))
# x1 = np.random.uniform(low=-1, high=1, size=(dim,))
vec_set = [x1]
c = 1
while len(vec_set) < n_vecs:
    new_v = get_anti_corr_vector(dim, vec_set, corr_limit)
    vec_set.append(new_v)
    np.shape(vec_set)
    c +=1
    print(c)
    fname = f'model/anticorr_ctxcs-n-{c}-dim-{dim}-corr%.2f.npy' % (corr_limit)
    np.save(fname, np.array(vec_set))


np.corrcoef(vec_set)
