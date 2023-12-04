'''fitting two periodic functions using an crp-rnn,
so this is a context dependent task
this is the basis of a bunch of learning hierachical seqs with crp-rnn
'''
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from utils import to_np, to_pth
from model import CRPLSTM as Agent

sns.set(style='white', context='poster')
torch.autograd.set_detect_anomaly(True)
# set seed
seed_val = 0
torch.manual_seed(seed_val)
np.random.seed(seed_val)

# params
n_epochs = 300
learning_rate = .03
noise_level = .01
dim_input = 1
dim_hidden = dim_context = 16
dim_output = 1

# make a sin wave
x = np.arange(-3 * np.pi, 3 * np.pi, .1)
y_sin = np.sin(x)
y_cos = np.cos(x)
context_sin = np.random.normal(0, 1, dim_context)
context_cos = np.random.normal(0, 1, dim_context)
# data processing
x = to_pth(x)
y_sin = to_pth(y_sin)
y_cos = to_pth(y_cos)
context_sin = to_pth(context_sin)
context_cos = to_pth(context_cos)
T = len(x)


def get_data():
    '''helper'''
    if np.random.uniform() > .5:
        y_ = y_sin + torch.randn(T) * noise_level
        context_ = context_sin + torch.randn(dim_context) * noise_level
    else:
        y_ = y_cos + torch.randn(T) * noise_level
        context_ = context_cos + torch.randn(dim_context) * noise_level
    return y_, context_


# init model
criterion = nn.MSELoss()
agent = Agent(
    dim_input, dim_hidden, dim_output, dim_context, output_format='continuous'
)
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

# train the model
for i in range(n_epochs):
    h_t = torch.zeros(1, 1, dim_hidden)
    c_t = torch.zeros(1, 1, dim_hidden)
    loss = 0
    y, context = get_data()
    for t in range(T):
        output, cache = agent.forward(
            x[t].view(1, 1, -1), h_t, c_t, context
        )
        [yhat_it, _, _, h_t, c_t] = output
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
    output, cache = agent.forward(
        x[t].view(1, 1, -1), h_t, c_t, context_sin
    )
    [yhat[t], _, _, h_t, c_t] = output

# plot it
f, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(x, y_sin, label='sin')
ax.plot(x, yhat, label='model fit')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
sns.despine()


# test the model
yhat = np.zeros(T,)
h_t = torch.zeros(1, 1, dim_hidden)
c_t = torch.zeros(1, 1, dim_hidden)
for t in range(T):
    output, cache = agent.forward(
        x[t].view(1, 1, -1), h_t, c_t, context_cos
    )
    [yhat[t], _, _, h_t, c_t] = output

# plot it
f, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(x, y_cos, label='cos')
ax.plot(x, yhat, label='model fit')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
sns.despine()
