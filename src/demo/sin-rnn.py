'''fitting a periodic function using an rnn
this is the basis of a bunch of learning hierachical seqs with crp-rnn
'''
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from utils import to_np, to_pth
from model import CRPLSTM_REG as Agent

sns.set(style='white', context='poster')
torch.autograd.set_detect_anomaly(True)
# set seed
seed_val = 0
torch.manual_seed(seed_val)
np.random.seed(seed_val)


# make a sin wave
x = np.arange(-3 * np.pi, 3 * np.pi, .1)
y = np.sin(x)

# # plot it
# f,ax =plt.subplots(1,1, figsize=(5,3))
# ax.plot(x,y)
# sns.despine()

x = to_pth(x)
y = to_pth(y)
T = len(x)

# model params
learning_rate = .01
dim_input = 1
dim_hidden = dim_context = 8
dim_output = 1


criterion = nn.MSELoss()
agent = Agent(dim_input, dim_hidden, dim_output, dim_context)
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

# train the model
n_epochs = 100
for i in range(n_epochs):
    h_t = torch.zeros(1, 1, dim_hidden)
    c_t = torch.zeros(1, 1, dim_hidden)
    loss = 0
    for t in range(T):
        context_t = torch.zeros(1, 1, dim_hidden)
        [yhat_it, h_t, c_t], cache = agent.forward(
            x[t].view(1, 1, -1), h_t, c_t, to_pth(context_t)
        )

        loss += criterion(yhat_it, y[t])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %3d | loss = %5.2f' % (i, to_np(loss)))


# test the model
yhat = np.zeros(T,)
h_t = torch.zeros(1, 1, dim_hidden)
c_t = torch.zeros(1, 1, dim_hidden)
for t in range(T):
    context_t = torch.zeros(1, 1, dim_hidden)
    [yhat[t], h_t, c_t], cache = agent.forward(
        x[t].view(1, 1, -1), h_t, c_t, to_pth(context_t)
    )


# plot it
f, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(x, y, label='ground truth')
ax.plot(x, yhat, label='model fit')
ax.set_xlabel('x')
ax.set_ylabel('y')
sns.despine()
