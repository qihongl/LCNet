import numpy as np
from copy import deepcopy
from utils import to_pth


class Polynomial():

    def __init__(
            self, n_terms=5, max_coeff_val=.5, max_degree=3, max_shift=3,
            xrange=1, xstep=.01, noise_level=0,
    ):
        self.n_terms = n_terms
        self.max_coeff_val = max_coeff_val
        self.max_degree = max_degree
        self.max_shift = max_shift
        self.xrange = xrange
        self.xstep = xstep
        self.noise_level = noise_level
        self.n_pts = int((xrange * 2) / xstep)
        self.reset()

    def reset(self):
        '''
        construct a funtion
        c1 x^0 + c2 x^1 + c3 x^2 + ...
        '''
        # get c1, c2, c3, ..., where c_i is a positive integer
        self.coeff = [
            np.random.uniform(-self.max_coeff_val, self.max_coeff_val)
            for i in range(self.n_terms)
        ]
        # self.shift = [
        #     np.random.uniform(-self.max_shift, self.max_shift)
        #     for i in range(self.n_terms)
        # ]
        # set d1, d2, d3, ... to be 0, 1, 2, ...
        # self.degree = [i for i in range(self.n_terms)]
        self.degree = [0] + [self.max_degree] * (self.n_terms - 1)

        # sample x from the real line
        self.x = np.arange(-self.xrange, self.xrange, self.xstep)
        # there is a option to change the number of terms you want to keep
        self.y = [self.sample_function(x_i) for x_i in self.x]

    def sample_function(self, x):
        y = 0
        for i, (c_i, d_i) in enumerate(zip(self.coeff, self.degree)):
            y += c_i * x ** d_i

        return y

    def sample(self, to_torch=False):
        noise = np.random.normal(0, self.noise_level, (len(self.x)))
        if to_torch:
            return to_pth(self.x), to_pth(self.y + noise)
        return self.x, self.y + noise


if __name__ == "__main__":
    '''how to use + sample a few poly'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    sns.set(style='white', palette='colorblind', context='poster')

    n_term_shared = 2
    n_term_idio = 2
    degree_shared = 2
    degree_idio = 3

    n_poly = 4
    xrange = 1

    # construct the polynomials
    np.random.seed(0)
    # get the shared poly
    spoly = Polynomial(
        n_terms=n_term_shared, max_degree=degree_shared, xrange=xrange
    )
    # get the idiosyncratic poly
    poly = [
        Polynomial(n_terms=n_term_idio, max_degree=degree_idio, xrange=xrange)
        for _ in range(n_poly)
    ]

    # sample from the shared components
    sx, sy = spoly.sample()
    # sample from idiosyncratic components
    ix, iy = np.zeros((n_poly, spoly.n_pts)), np.zeros((n_poly, spoly.n_pts))
    for i in range(n_poly):
        ix[i, :], iy[i, :] = poly[i].sample()
    # sum to get y
    y = iy + np.tile(sy, (n_poly, 1))

    # plot the shared and the composed polys
    f, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axes[0].plot(sx, sy, color='k')
    for i in range(n_poly):
        axes[0].plot(sx, iy[i, :])
        axes[1].plot(sx, y[i, :])
    for ax in axes:
        ax.set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Shared / idiosyncratic components')
    axes[0].legend(['Shared'])
    axes[1].set_title('Sum')
    sns.despine()
    f.tight_layout()
