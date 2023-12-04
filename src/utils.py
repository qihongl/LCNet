import os
import sys
import warnings
import torch
import pickle
import numpy as np

from itertools import product
from scipy.stats import sem
from torch.nn.functional import smooth_l1_loss
from copy import deepcopy
from scipy.linalg import qr


eps = np.finfo(np.float32).eps.item()

def is_even(number):
    if number % 2 == 0:
        return 1
    return 0

def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.tensor(np_array).type(pth_dtype)


def to_sqpth(np_array, pth_dtype=torch.FloatTensor):
    return torch.squeeze(to_pth(np_array, pth_dtype=pth_dtype))


def to_np(torch_tensor):
    return torch_tensor.data.numpy()


def to_sqnp(torch_tensor, dtype=np.float):
    return np.array(np.squeeze(to_np(torch_tensor)), dtype=dtype)


def enumerated_product(*args):
    # https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))

def stable_softmax(x, beta=1, subtract_max=True):
    assert beta > 0
    if subtract_max:
        x -= max(x)
    # apply temperture
    z = x / beta
    return np.exp(z) / (np.sum(np.exp(z)) + 1e-010)

def ignore_warnings():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

def ortho_proj(u, v):
    '''compute orthogonal project of u on v'''
    v_norm = np.sqrt(np.sum(v**2))
    return (np.dot(u, v)/v_norm**2)*v

def compute_returns(rewards, gamma=0, normalize=False):
    """compute return in the standard policy gradient setting.

    Parameters
    ----------
    rewards : list, 1d array
        immediate reward at time t, for all t
    gamma : float, [0,1]
        temporal discount factor
    normalize : bool
        whether to normalize the return
        - default to false, because we care about absolute scales

    Returns
    -------
    1d torch.tensor
        the sequence of cumulative return

    """
    # rewards = to_pth(rewards)
    # compute cumulative discounted reward since t, for all t
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # normalize w.r.t to the statistics of this trajectory
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    # from utils.utils import to_sqnp
    # np.set_printoptions(precision=1)
    # print(to_sqnp(returns))
    return returns


def compute_a2c_loss(probs, values, returns, use_V=True):
    """compute the objective node for policy/value networks

    Parameters
    ----------
    probs : list
        action prob at time t
    values : list
        state value at time t
    returns : list
        return at time t

    Returns
    -------
    torch.tensor, torch.tensor
        Description of returned object.

    """
    policy_grads, value_losses = [], []
    for prob_t, v_t, R_t in zip(probs, values, returns):
        if use_V:
            A_t = R_t - v_t.item()
            value_losses.append(
                smooth_l1_loss(torch.squeeze(v_t), torch.squeeze(R_t))
            )
        else:
            A_t = R_t
            value_losses.append(torch.FloatTensor(0).data)
        # accumulate policy gradient
        policy_grads.append(-prob_t * A_t)
    policy_gradient = torch.stack(policy_grads).sum()
    value_loss = torch.stack(value_losses).sum()
    return policy_gradient, value_loss


def append_extra(x_it_, scalar_list):
    for s in scalar_list:
        x_it_ = torch.cat(
            [x_it_, s.type(torch.FloatTensor).view(tensor_length(s))]
        )
    return x_it_


def to_1d_tensor(scalar_list):
    return torch.cat(
        [s.type(torch.FloatTensor).view(tensor_length(s)) for s in scalar_list]
    )


def tensor_length(tensor):
    if tensor.dim() == 0:
        length = 1
    elif tensor.dim() > 1:
        raise ValueError('length for high dim tensor is undefined')
    else:
        length = len(tensor)
    return length


def get_reward(chosen_arm, correct_arm):
    r_t = 0
    if to_np(chosen_arm) == to_np(correct_arm):
        r_t = 1
    return torch.from_numpy(np.array(r_t)).type(torch.FloatTensor).data


def pickle_save_dict(input_dict, save_path):
    """Save the dictionary

    Parameters
    ----------
    input_dict : type
        Description of parameter `input_dict`.
    save_path : type
        Description of parameter `save_path`.

    """
    with open(save_path, 'wb') as handle:
        pickle.dump(input_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_dict(fpath):
    """load the dict

    Parameters
    ----------
    fpath : type
        Description of parameter `fpath`.

    Returns
    -------
    type
        Description of returned object.

    """
    return pickle.load(open(fpath, "rb"))


def binarize_array(arr):
    arr[arr > 0] = 1
    arr[arr <= 0] = 0
    return arr


def sample_binary_array(array_dim, n_arrays):
    arr = np.array([np.random.normal(size=array_dim) for _ in range(n_arrays)])
    barr = binarize_array(arr)
    return np.array(barr, dtype=np.int)


def flip(x):
    if x == 0:
        return 1
    elif x == 1:
        return 0
    else:
        raise ValueError('x to be fliped, must be 0 or 1')


def flip_bits(input_percept, noise_level):
    percept_dim = len(input_percept)
    noisy_percept = deepcopy(input_percept)
    # decide # bit to be flipped
    n_bit_flip = int(np.round(noise_level * percept_dim))
    # decide which bits to flip
    bit_flip = np.random.choice(range(percept_dim), n_bit_flip, replace=False)
    # flip them
    for b_i in bit_flip:
        noisy_percept[b_i] = flip(noisy_percept[b_i])
    return noisy_percept


def count_cluster_assignment(cluster_assignments, n_clusters):
    assignm_dict = {i: [] for i in range(n_clusters)}
    for p_i_, c_i_ in cluster_assignments:
        assignm_dict[c_i_].append(p_i_)
    return assignm_dict


def mapping_count(assignm_dict, n_clusters):
    map_count = np.zeros(n_clusters,)
    for c_i_, p_i_s in assignm_dict.items():
        map_count[c_i_] = len(set(p_i_s))
    return map_count


def cosine_sim(mat1, mat2=None):
    '''
    mat is a # items by # dim matrix
    '''
    if mat2 is None:
        mat2 = mat1
    assert np.shape(mat1)[1] == np.shape(mat2)[1]
    m = np.shape(mat1)[0]
    n = np.shape(mat2)[0]
    sim_mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            # compute cosine similarity
            sim_mat[i,j] = np.dot(mat1[i,:], mat2[j,:]) / (np.linalg.norm(mat1[i,:]) * np.linalg.norm(mat2[j,:]))
    return sim_mat

def random_ortho_mat(dim):
    Q, _ = qr(np.random.randn(dim, dim))
    return Q


def compute_smth_mean(x_i, w_cur=.5):
    event_len = len(x_i)
    smth_x = np.zeros((event_len+1, len(x_i[0])))
    for t in range(event_len):
        smth_x[t+1] = compute_smth_mean_t(smth_x[t], x_i[t], w_cur=w_cur)
    return smth_x[1:, :]

def compute_smth_mean_t(x_prev, x_cur, w_cur=.5):
    assert np.shape(x_prev) == np.shape(x_cur)
    return x_prev * (1-w_cur) + x_cur * w_cur


def _test_compute_smth_mean():
    '''how to compute smoothed mean online'''
    # assume there are n points
    npts = 9
    x = np.random.normal(loc=0, scale=1, size = (npts, 1))

    # preallocate for n+1 points, the 1st 0 vec will be used as the starting pt
    sx = np.zeros(npts+1,)
    for t in range(npts):
        sx[t+1] = compute_smth_mean_t(sx[t], x[t, 0])
    # at the end, the 1: points will be the smooted inputs
    sx_test = compute_smth_mean(x)
    close_enough = np.allclose(sx[1:], sx_test.reshape(-1))
    if not close_enough:
        raise ValueError()



def find_loc_longest_zeros(seq):
    '''
    reference: https://stackoverflow.com/questions/40166522/find-longest-sequence-of-0s-in-the-integer-list

    test case:
    seq = [1,2,0,0,3,4,5,-1,0,2,-1,-3,0,0,0,0,0,0,0,0,2,-3,-4,-5,0,0,0]
    find_loc_longest_zeros(seq)
    print("The longest sequence of 0's is "+str(prev))
    print("index start at: "+ str(indexend-prev))
    print("index ends at: "+ str(indexend-1))

    output:
    The longest sequence of 0's is 8
    index start at: 12
    index ends at: 19
    '''
    count = 0
    prev = 0
    indexend = 0
    indexcount = 0
    for i in range(0,len(seq)):
        if seq[i] == 0:
            count += 1
            indexcount = i
        else:
            if count > prev:
                prev = count
                indexend = i
                count = 0

    if count > prev:
        prev = count
        indexend = indexcount
    # compute useful info
    len_longest_zero_seq = prev
    loc_longest_zero_seq = indexend - prev
    end_longest_zero_seq = indexend-1
    return loc_longest_zero_seq




if __name__ == "__main__":
    '''how to use'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', palette='colorblind', context='poster')

    block_size = 16
    n_samples = 16
    ctx_id_seq = get_blocked_sequence(block_size, n_samples)
    plt.plot(ctx_id_seq)

    # plot param
    np.random.seed(1)
    cp = sns.color_palette(n_colors = 3)
    head_width = .1
    head_length = .12
    # sample random u and v
    dim = 2
    scale = 1
    u = np.random.normal(size=(dim), scale=scale)
    v = np.random.normal(size=(dim), scale=scale)
    # projection
    upv = ortho_proj(u, v)

    # plot the vectors
    f, ax = plt.subplots(1,1, figsize = (8, 8))

    origin = np.array([0, 0])
    ax.arrow(*origin, u[0], u[1], label = 'u', color=cp[0],  head_width=head_width, head_length=head_length)
    ax.arrow(*origin, v[0], v[1], label = 'v', color=cp[1],  head_width=head_width, head_length=head_length)
    ax.arrow(*origin, upv[0], upv[1], label = 'the projection of u on v', color=cp[2],  head_width=head_width, head_length=head_length)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')

    ax.legend()
    sns.despine()

    _test_compute_smth_mean()

    # how to make block and interleaved curricumulm
    block_size = 64
    n_samples = block_size
    n_tasks = 2
    seq1 = get_blocked_sequence(block_size, n_samples, n_tasks=n_tasks)
    seq2 = get_blocked_sequence(1, n_samples, n_tasks=n_tasks)
    seq3 = get_random_seq(block_size*n_tasks, n_tasks=n_tasks)
    # blocked-1st
    seq = np.concatenate([seq1] * 1 + [seq2] * 1 + [seq3] * 2)
    # interleaved-1st
    seq = np.concatenate([seq2] * 1 + [seq1] * 1 + [seq3] * 2)

    plt.plot(seq)
    print(len(seq1))
    print(len(seq2))
    print(len(seq3))
    print(len(seq))




    # print(Q.dot(Q.T))
    # cosine_sim(Q)
