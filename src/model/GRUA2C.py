"""
a gru-a2c
the gru uses the implementation from:
https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
"""
import math
import torch
import torch.nn as nn
from models.A2C import A2C_linear
# from models.A2C import A2C
from models._rl_helpers import pick_action


class GRU(nn.Module):

    """
    An implementation of GRU.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, ctx_wt, beta=1, bias=True):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.beta = beta
        self.bias = bias
        self.i2h = nn.Linear(input_dim, 3 * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.a2c = A2C_linear(self.hidden_dim, self.output_dim)
        # self.a2c = A2C(self.hidden_dim, self.hidden_dim, self.output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def update_beta(self, new_beta):
        self.beta = new_beta

    def forward(self, x, hidden):
        x = x.view(1, -1)

        gate_x = self.i2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 0)
        h_r, h_i, h_n = gate_h.chunk(3, 0)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        h_t = newgate + inputgate * (hidden - newgate)

        a_t, prob_a_t, v_t = self.get_output(h_t)

        output = [a_t, prob_a_t, v_t, h_t]
        cache = [resetgate, inputgate, newgate]
        return output, cache


    def get_output(self, h_t):
        '''generate the rl output
        '''
        pi_a_t, v_t = self.a2c.forward(h_t, self.beta)
        # pick an action
        a_t, prob_a_t = pick_action(pi_a_t)
        return a_t, prob_a_t, v_t

    # @torch.no_grad()
    def add_normal_noise(self, pattern, scale=.1):
        return pattern + torch.randn(pattern.size()) * scale

    @torch.no_grad()
    def get_zero_states(self):
        h_0 = torch.zeros(1, 1, self.hidden_dim)
        return h_0
