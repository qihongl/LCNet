"""
An implementation of Context dependent GRU.
the gru uses the implementation from:
https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
"""
import math
import torch
import torch.nn as nn
import numpy as np

class CGRU(nn.Module):

    def __init__(
        self, input_dim, hidden_dim, output_dim, context_dim, ctx_wt, bias=True,
        sigmoid_output=True, dropout_rate=.5
        ):
        super(CGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.bias = bias
        self.ctx_wt = ctx_wt
        # weights
        self.i2h = nn.Linear(input_dim, 3 * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.h2o = nn.Linear(hidden_dim, output_dim, bias=bias)
        # context layer
        self.ci2h = nn.Linear(context_dim, 3 * hidden_dim, bias=bias)
        self.ch2h = nn.Linear(context_dim, 3 * hidden_dim, bias=bias)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.i2h_dropout = nn.Dropout(dropout_rate)
            self.ci2h_dropout = nn.Dropout(dropout_rate)
            self.h2o_dropout = nn.Dropout(dropout_rate)
        # miscs
        self.sigmoid_output = sigmoid_output
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        # std = 1.0 / self.hidden_dim
        for w in self.parameters():
            w.data.uniform_(-std, std)
        # for layer in range(num_layers):
        #     for weight in rnn._all_weights[layer]:
        #         if "weight" in weight:
        #             nn.init.xavier_uniform_(getattr(rnn,weight))
        #         if "bias" in weight:
        #             nn.init.uniform_(getattr(rnn,weight))

        # for name, wts in self.named_parameters():
        #     if 'weight' in name:
        #         torch.nn.init.xavier_uniform_(wts)
        #     elif 'bias' in name:
        #         torch.nn.init.uniform_(wts, 0)


    # def reset_parameters(self):
    #     for name, wts in self.named_parameters():
    #         if 'weight' in name:
    #             torch.nn.init.orthogonal_(wts)
    #         elif 'bias' in name:
    #             torch.nn.init.constant_(wts, 0)

    def forward(self, x, hidden, context_t=None):
        x = x.view(1, -1)
        gate_x = self.i2h(x) * (1 - self.ctx_wt) + self.ci2h(context_t) * self.ctx_wt
        gate_h = self.h2h(hidden) * (1 - self.ctx_wt) + self.ch2h(context_t) * self.ctx_wt
        # gate_x = self.i2h(x) + self.ci2h(context_t)
        # gate_h = self.h2h(hidden) + self.ch2h(context_t)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 0)
        h_r, h_i, h_n = gate_h.chunk(3, 0)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        h_t = newgate + inputgate * (hidden - newgate)
        if self.dropout_rate > 0:
            h_t = self.h2o_dropout(h_t)
        a_t = self.h2o(h_t)
        if self.sigmoid_output:
            a_t = a_t.sigmoid()

        output = [a_t, h_t]
        cache = [resetgate, inputgate, newgate]
        return output, cache


    def forward_nograd(self, x_t, h, context_t=None):
        with torch.no_grad():
            [a_t, h_t], _ = self.forward(x_t, h,context_t=context_t)
        return a_t

    def try_all_contexts(self, y_t, x_t, h_t, contexts):
        # loop over all ctx ...
        # ... AND the zero context at index 0
        n_contexts = len(contexts)
        match = [None] * (n_contexts)
        for k in range(n_contexts):
            yhat_k = self.forward_nograd(x_t, h_t, contexts[k])
            match[k] = 1 - np.abs(torch.squeeze(yhat_k).numpy() - y_t)
        return np.array(match)

    # def try_all_contexts(self, y_t, x_t, h_t, contexts, criterion):
    #     # loop over all ctx ...
    #     # ... AND the zero context at index 0
    #     n_contexts = len(contexts)
    #     match = [None] * (n_contexts)
    #     for k in range(n_contexts):
    #         yhat_k = self.forward_nograd(x_t, h_t, contexts[k])
    #         match[k] = 10 - criterion(y_t, torch.squeeze(yhat_k))
    #         # torch.squeeze(yhat_k)[torch.argmax(y_t)].numpy()
    #     return np.array(match)

    def get_kaiming_states(self):
        return nn.init.kaiming_uniform_(torch.empty(1, 1, self.hidden_dim))

    @torch.no_grad()
    def get_zero_states(self):
        h_0 = torch.zeros(1, 1, self.hidden_dim)
        return h_0

    @torch.no_grad()
    def get_rand_states(self, scale=.1):
        h_0 = torch.randn(1, 1, self.hidden_dim) * scale
        return h_0

def freeze_layer(layer_to_freeze, model, verbose=False):
    # layer_to_freeze = 'ci2h'
    layer_found = False
    for name, param in model.named_parameters():
        if layer_to_freeze in name:
            layer_found = True
            if param.requires_grad:
                param.requires_grad = False
            else:
                print(f'layer already freezed: {name}')
                if verbose:
                    print(f'freeze layer: {name}')
    if not layer_found:
        raise ValueError(f'layer {name} not found')
    return model

def get_weights(layer_name, model, to_np=True):
    weights = {}
    for name, param in model.named_parameters():
        if layer_name in name:
            w_ = param.data.numpy() if to_np else param.data
            weights[name] = w_
    return weights
