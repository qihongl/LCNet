'''train a regular nn on two bandits -- see if ...
1. is there a development of certainty representation, which can be used to trigger CRP
2. is h_t different across bandits, which is needed for CRP to do the right splitting
'''
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from utils import to_np, to_pth, to_1d_tensor
from utils import compute_a2c_loss, compute_returns
from model import CRPLSTM as Agent
from task import SimpleTwoArmBandit

sns.set(style='white', palette='colorblind', context='talk')
cpal = sns.color_palette()
print(f'GPU avail: {torch.cuda.is_available()}')

# set seed
seed_val = 0
torch.manual_seed(seed_val)
np.random.seed(seed_val)

# condition params
use_ctx = False
# training param
n_epochs = 1000
lr = 1e-3
# network param
input_dim = 2  # prev a, prev r
output_dim = 3  # need 2d-vector to represent a choice in a 2 arm bandit
hidden_dim = 16
dim_context = 0

# init model
agent = Agent(input_dim, hidden_dim, output_dim, dim_context, use_ctx=False)
optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

# init 2 deterministic bandit (i.e. b([0,1]) vs b([1,0]))
ps = [1, 0]
reward = 1
penalty = -1
n_tps = 20
bandits = [SimpleTwoArmBandit(p, reward=reward, penalty=penalty) for p in ps]


def sample_bandit_ordering(n_tps, gap=3, n_bandits=2):

    bandit_switch_point = np.random.choice(
        np.arange(n_tps // 2 - gap, n_tps // 2 + gap)
    )
    bandit_length = [bandit_switch_point, n_tps - bandit_switch_point]
    bandit_order = np.random.permutation(n_bandits)
    bandit_index = np.concatenate(
        [[bandit_order[i]] * bandit_length[i] for i in range(n_bandits)]
    )
    return bandit_index


def run(learning=True):
    # train the model on the two bandit tasks
    log_a_ij = np.zeros(n_tps,)
    log_c_ij = np.zeros((n_tps, hidden_dim))
    log_h_ij = np.zeros((n_tps, hidden_dim))

    # choose the bandit order
    bandit_index = sample_bandit_ordering(n_tps)
    probs, rewards, values = [], [], []
    a_t = torch.tensor(2)
    r_t = torch.tensor(0).type(torch.FloatTensor)
    h_t, c_t = agent.get_init_states()
    for t in range(n_tps):
        # forward
        x_t = to_1d_tensor([a_t, r_t])
        [a_t, pi_a_t, v_t, h_t, c_t], gate = agent.forward(
            x_t.view(1, 1, -1), h_t, c_t,
        )
        # compute reward
        r_t = to_pth(bandits[bandit_index[t]].get_reward(a_t))
        # log info
        rewards.append(r_t)
        values.append(v_t)
        probs.append(pi_a_t)
        log_a_ij[t] = to_np(a_t)
        log_c_ij[t] = to_np(c_t)
        log_h_ij[t] = to_np(h_t)
    # compute loss
    returns = compute_returns(rewards, normalize=False)
    loss_actor, loss_critic = compute_a2c_loss(
        probs, values, returns)
    loss = loss_actor + loss_critic
    # update weights
    if learning:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
        optimizer.step()
    cache = [log_a_ij, log_c_ij, log_h_ij, bandit_index]
    return returns, cache


'''train the model'''
log_return = np.zeros(n_epochs)
log_a = np.zeros((n_epochs, n_tps))
log_c = np.zeros((n_epochs, n_tps, hidden_dim))
log_h = np.zeros((n_epochs, n_tps, hidden_dim))
log_bandit_id = np.zeros((n_epochs, n_tps))
for i in range(n_epochs):
    returns, cache = run()
    log_return[i] = np.mean(to_np(returns))
    log_a[i], log_c[i], log_h[i], log_bandit_id[i] = cache
    if i % 100 == 0:
        msg = 'Epoch %3d | reward = %6.4f' % (i, log_return[i])
        print(msg)
