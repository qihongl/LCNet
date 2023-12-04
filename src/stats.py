import numpy as np
from scipy.stats import sem
# from scipy.special import kl_div
from sklearn.metrics import accuracy_score
from sklearn import metrics

def compute_stats(arr, axis=0, n_se=2, use_se=True):
    """compute mean and errorbar w.r.t to SE
    Parameters
    ----------
    arr : nd array
        data
    axis : int
        the axis to do stats along with
    n_se : int
        number of SEs
    Returns
    -------
    (n-1)d array, (n-1)d array
        mean and se
    """
    mu_ = np.mean(arr, axis=axis)
    if use_se:
        er_ = sem(arr, axis=axis) * n_se
    else:
        er_ = np.std(arr, axis=axis)
    return mu_, er_


def moving_average(x, winsize):
    return np.convolve(x, np.ones(winsize), 'valid') / winsize


def compute_norm_pe(x, m1, m2, v):
    '''compute prediction error and normalization factor under two gaussians'''
    v1 = v2 = v
    squared_error = ((x - m2) ** 2) / v2 - ((x - m1) ** 2) / v1
    norm_factor = 2 * np.log(v1 / v2)
    return squared_error, norm_factor


def ber_kl_div(p, q, small_num = 1e-6):
    '''compute bernuli KL divergence
    p, q: 1-D float
    '''
    def in_0to1(pp):
        if pp == 0:
            pp += small_num
        elif pp == 1:
            pp -= small_num
        return pp
    # bound p and q to (0, 1)
    p = in_0to1(p)
    q = in_0to1(q)
    return p * (np.log(p/q)) + (1-p) * (np.log((1-p)/(1-q)))

# def purity_score(y_true, y_pred):
#     """Purity score
#         Args:
#             y_true(np.ndarray): n*1 matrix Ground truth labels
#             y_pred(np.ndarray): n*1 matrix Predicted clusters
#
#         Returns:
#             float: Purity score
#     """
#     # matrix which will hold the majority-voted labels
#     y_voted_labels = np.zeros(y_true.shape)
#     # Ordering labels
#     # Labels might be missing e.g with set like 0,2 where 1 is missing
#     # First find the unique labels, then map the labels to an ordered set
#     # 0,2 should become 0,1
#     labels = np.unique(y_true)
#     ordered_labels = np.arange(labels.shape[0])
#     for k in range(labels.shape[0]):
#         y_true[y_true == labels[k]] = ordered_labels[k]
#     # Update unique labels
#     labels = np.unique(y_true)
#     # We set the number of bins to be n_classes+2 so that
#     # we count the actual occurence of classes between two consecutive bins
#     # the bigger being excluded [bin_i, bin_i+1[
#     bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)
#
#     for cluster in np.unique(y_pred):
#         hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
#         # Find the most present label in the cluster
#         winner = np.argmax(hist)
#         y_voted_labels[y_pred == cluster] = winner
#
#     return accuracy_score(y_true, y_voted_labels)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
