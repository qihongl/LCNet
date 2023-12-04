'''define prediction error in the case of 2 gaussians'''
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns

sns.set(style='white', palette='colorblind', context='talk')
cp = sns.color_palette(n_colors=2)
alpha = .7
np.random.seed(0)

'''helper func '''
def compute_norm_pe(x, m1, m2, v):
    '''compute prediction error and normalization factor under two gaussians
    decide in favor of h1 when:
    log P(x|h1) + log P(h1) - log P(x|h2) - log P(h2) > 0
    let's assume that the prior is uniform, hence P(h1) = P(h2) = 0.5, and the decision rule is:
    log P(x|h1) - log P(x|h2) > 0
    plug in the Gaussian pdf and do a bit of manipulation, you get:
    (x-m2)^2/v2 - (x-m1)^2/v1> 2*log(v1/v2)
    '''
    v1 = v2 = v
    squared_prediction_error = ((x - m2) ** 2) / v2 - ((x - m1) ** 2) / v1
    log_var_ratio = 2 * np.log(v1 / v2)
    return squared_prediction_error, log_var_ratio

def classify(x, m1, m2, v):
    squared_prediction_error, log_var_ratio = compute_norm_pe(x, m1, m2, v)
    spe = squared_prediction_error - log_var_ratio
    if spe > 0:
        cls = 0
    else:
        cls = 1
    return cls, spe

def compute_stats(arr, axis=0, n_se=2):
    mu_ = np.mean(arr, axis=axis)
    er_ = sem(arr, axis=axis) * n_se
    return mu_, er_

'''setup'''
# set two gaussian
m1 = -1
m2 = 1

# take some samples
n_samples = 30
samples = np.random.uniform(low=-6, high=6, size=n_samples)

# plot the two gaussians
f, axes = plt.subplots(2, 2, figsize = (20, 8), sharex=True)
vs = [1, 3]
for j, vj in enumerate(vs):
    v1 = v2 = vj
    # compute the PE and classes
    cls = np.zeros(len(samples), dtype=np.int)
    spe = np.zeros(len(samples))
    for i, x in enumerate(samples):
        cls[i], spe[i] = classify(x, m1, m2, vj)

    halfrange = 10
    xrange = np.linspace(-halfrange, halfrange, 100)
    axes[0, j].plot(xrange, stats.norm.pdf(xrange, m1, v1), color = cp[0], alpha=alpha)
    axes[0, j].plot(xrange, stats.norm.pdf(xrange, m2, v2), color = cp[1], alpha=alpha)
    axes[0, j].set_xlabel('x')
    axes[0, j].set_ylabel('Density')
    axes[0, j].set_title(f'gaussian, mu1 = {m1}, mu2 = {m2}, var1 = var2 = {v1}')
    axes[0, j].set_ylim([None, .45])
    axes[0, j].axvline((m1 + m2)/ 2, color='grey', linestyle='--')
    axes[0, j].axvline(m1, color=cp[0], linestyle='--')
    axes[0, j].axvline(m2, color=cp[1], linestyle='--')
    sns.despine()

    # plot the classes
    for c, x in zip(cls, samples):
        axes[0, j].scatter(x, 0, color = cp[c])

    # plot the error
    axes[1, j].stem(samples, spe, linefmt=None, markerfmt=None, basefmt=None)
    axes[1, j].set_xlabel('x')
    axes[1, j].set_ylabel('evidence of m1')
    axes[1, j].set_ylim([-25, 25])
    axes[1, j].axvline((m1 + m2)/ 2, color='grey', linestyle='--')
    axes[1, j].axvline(m1, color=cp[0], linestyle='--')
    axes[1, j].axvline(m2, color=cp[1], linestyle='--')
    sns.despine()

n_data = 100
n_sims = 20
true_var = 3
var_over_n_data = np.zeros((n_sims, n_data-1))
for i in range(n_sims):
    data = np.random.normal(loc = 0, scale = true_var, size=n_data)
    var_over_n_data[i] = np.array([np.std(data[:i+1]) for i in range(n_data-1)])

# var = np.zeros(n_data)
# for n in range(n_data):
#     var[n] = np.sqrt(np.sum((data[:n] - np.mean(data[:n]))**2) / (n) )

mu, se = compute_stats(var_over_n_data)

f, ax = plt.subplots(1, 1, figsize = (10, 6))
ax.errorbar(x=range(n_data-1),y=mu,yerr=se)

ax.set_xlabel('N samples')
ax.axhline(true_var, linestyle = '--', color = 'red')
ax.set_ylabel('estimated var')
sns.despine()
