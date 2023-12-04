"""
a lstm-a2c with context
context input is added to the linear layer of A2C
"""
import torch
import torch.nn as nn
import numpy as np
from model.A2C import A2C_linear
# from model.A2C import A2C


# constants
N_GATES = 4
# output format is either discrete (e.g. action selection) or continuous
OUTPUT_FORMAT = ['discrete', 'continuous']


class CRPLSTM(nn.Module):

    def __init__(
            self, input_dim, hidden_dim, output_dim, context_dim=0,
            bias=True, output_format='discrete', use_ctx=True, ctx_wt=.5, sigmoid_output=False,
    ):
        super(CRPLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.bias = bias
        self.use_ctx = use_ctx
        self.ctx_wt = ctx_wt
        self.sigmoid_output = sigmoid_output

        n_units_preact = (N_GATES + 1) * hidden_dim
        # input-hidden weights
        self.i2h = nn.Linear(input_dim, n_units_preact, bias=bias)
        # hidden-hidden weights
        self.h2h = nn.Linear(hidden_dim, n_units_preact, bias=bias)
        # context-hidden weights
        if use_ctx:
            self.c2h = nn.Linear(context_dim, n_units_preact, bias=bias)
        # set the output module depending on the task requirement
        self.set_output_module(output_format)
        # init
        self.reset_parameter()

    def set_output_module(self, output_format):
        assert output_format in OUTPUT_FORMAT
        self.output_format = output_format
        if output_format == 'discrete':
            # policy net
            self.a2c = A2C_linear(self.hidden_dim, self.output_dim)
            # self.a2c = A2C(self.hidden_dim, self.hidden_dim, self.output_dim)
        else:
            # hidden-output weights
            self.h2o = nn.Linear(
                self.hidden_dim, self.output_dim, bias=self.bias
            )

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)

    def forward(self, x_t, h, c, context_t=None):
        # x_t = x_t.view(1, 1, -1)
        # unpack activity
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x_t = x_t.view(x_t.size(1), -1)
        # transform the input info
        preact = self.i2h(x_t) + self.h2h(h)
        if self.use_ctx:
            preact = preact * (1 - self.ctx_wt) + self.c2h(context_t) * self.ctx_wt
            # preact += self.c2h(context_t)
        # get all gate values
        gates = preact[:, : N_GATES * self.hidden_dim].sigmoid()
        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, :self.hidden_dim]
        i_t = gates[:, self.hidden_dim:2 * self.hidden_dim]
        o_t = gates[:, 2 * self.hidden_dim:3 * self.hidden_dim]
        r_t = gates[:, -self.hidden_dim:]
        # stuff to be written to cell state
        c_t_new = preact[:, N_GATES * self.hidden_dim:].tanh()
        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(f_t, c) + torch.mul(i_t, c_t_new)
        # get gated hidden state from the cell state
        h_t = torch.mul(o_t, c_t.tanh())
        # get output
        a_t, prob_a_t, v_t = self.get_output(h_t)
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        # fetch activity
        output = [a_t, prob_a_t, v_t, h_t, c_t]
        cache = [f_t, i_t, o_t, r_t]
        return output, cache

    def forward_nograd(self, x_t, h, c, context_t=None):
        with torch.no_grad():
            [a_t, prob_a_t, v_t, h_t, c_t], _ = self.forward(
                x_t, h, c, context_t=context_t)
        return a_t

    # def try_all_contexts(self, y_t, x_t, h_t, c_t, contexts, criterion):
    #     # loop over all ctx ...
    #     # ... AND the zero context at index 0
    #     n_contexts = len(contexts)
    #     losses = [None] * (n_contexts)
    #     for k in range(n_contexts):
    #         yhat_k = self.forward_nograd(x_t, h_t, c_t, contexts[k])
    #         losses[k] = criterion(torch.squeeze(yhat_k), y_t)
    #     return losses

    # def try_all_contexts(self, y_t, x_t, h_t, c_t, contexts):
    #     # loop over all ctx ...
    #     # ... AND the zero context at index 0
    #     n_contexts = len(contexts)
    #     match = [None] * (n_contexts)
    #     for k in range(n_contexts):
    #         yhat_k = self.forward_nograd(x_t, h_t, c_t, contexts[k])
    #         match[k] = torch.squeeze(yhat_k)[torch.argmax(y_t)].numpy()
    #     return np.array(match)


    def try_all_contexts(self, y_t, x_t, h_t, c_t, contexts):
        # loop over all ctx ...
        # ... AND the zero context at index 0
        n_contexts = len(contexts)
        match = [None] * (n_contexts)
        for k in range(n_contexts):
            yhat_k = self.forward_nograd(x_t, h_t, c_t, contexts[k])
            match[k] = 1 - np.abs(torch.squeeze(yhat_k).numpy() - y_t)
        return np.array(match)



    def get_output(self, h_t):
        '''generate the output depending on the task requirement
        if discrete - rl
        if continuous, do regression
        '''
        if self.output_format == 'discrete':
            # policy
            pi_a_t, v_t = self.a2c.forward(h_t)
            # pick an action
            a_t, prob_a_t = pick_action(pi_a_t)
        else:
            a_t = self.h2o(h_t)
            prob_a_t, v_t = None, None
            if self.sigmoid_output:
                a_t = a_t.sigmoid()
                # a_t = torch.clip(a_t, min=0, max=1)
        return a_t, prob_a_t, v_t

    def get_init_states(self, scale=.1):
        h_0 = torch.randn(1, 1, self.hidden_dim) * scale
        c_0 = torch.randn(1, 1, self.hidden_dim) * scale
        # h_0 = torch.zeros(1, 1, self.hidden_dim)
        # c_0 = torch.zeros(1, 1, self.hidden_dim)
        return h_0, c_0

    # @torch.no_grad()
    def get_zero_states(self):
        h_0 = torch.zeros(1, 1, self.hidden_dim)
        return h_0

    # @torch.no_grad()
    def get_rand_states(self, scale=.1):
        h_0 = torch.randn(1, 1, self.hidden_dim) * scale
        return h_0


def pick_action(action_distribution):
    """action selection by sampling from a multinomial.

    Parameters
    ----------
    action_distribution : 1d torch.tensor
        action distribution, pi(a|s)

    Returns
    -------
    torch.tensor(int), torch.tensor(float)
        sampled action, log_prob(sampled action)

    """
    m = torch.distributions.Categorical(action_distribution)
    a_t = m.sample()
    log_prob_a_t = m.log_prob(a_t)
    return a_t, log_prob_a_t
