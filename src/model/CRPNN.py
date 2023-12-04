import torch
import torch.nn as nn


class CRPNN(nn.Module):
    '''
    NN with 1 RELU hidden layer, and CRP sends c_k to the hidden layer,
    where c_k is the vector representation of the k-th context/task
    '''

    def __init__(self, dim_input, dim_hidden, dim_output, dim_context):

        super(CRPNN, self).__init__()
        self.nonlin = nn.ReLU()
        self.fc1 = nn.Linear(dim_input, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden + dim_context, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_output)

    def forward(self, x, c_k):
        preact1 = self.fc1(x)
        h1 = self.nonlin(preact1)
        hc1 = torch.cat([h1, c_k.view(1, -1)], dim=-1)
        preact2 = self.fc2(hc1)
        h2 = self.nonlin(preact2)
        out = self.fc3(h2)
        return out


class SimpleNN(nn.Module):
    '''
    NN with 1 RELU hidden layer
    '''

    def __init__(self, dim_input, dim_hidden, dim_output):

        super(SimpleNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(dim_input, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        preact1 = self.fc1(x)
        h1 = self.relu(preact1)
        preact2 = self.fc2(h1)
        h2 = self.relu(preact2)
        out = self.fc3(h2)
        return out


class SEM():
    def __init__(self, dim_input, dim_hidden, dim_output, lr):
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.lr = lr
        self.reset()

    def reset(self):
        self.models = []
        self.optims = []
        self.n_contexts = 0

    def add_model(self):
        new_model = SimpleNN(self.dim_input, self.dim_hidden, self.dim_output)
        self.models.append(new_model)
        self.optims.append(torch.optim.Adam(
            new_model.parameters(), lr=self.lr)
        )
        self.n_contexts += 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    dim_input = 1
    dim_hidden = 3
    dim_context = 3
    dim_output = 1

    x = torch.zeros(1, 1)
    c_k = torch.randn(1, 1, dim_context)

    crpnn = CRPNN(dim_input, dim_hidden, dim_output, dim_context)
    yhat = crpnn.forward(x, c_k)
    print(yhat)

    simplenn = SimpleNN(dim_input, dim_hidden, dim_output)
    yhat = simplenn.forward(x)
    print(yhat)
