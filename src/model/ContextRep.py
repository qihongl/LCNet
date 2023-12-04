import numpy as np


class ContextRep():
    def __init__(self, context_dim, threshold=.3, n_contexts=100):
        self.context_dim = context_dim
        self.n_contexts = n_contexts
        self.threshold = threshold
        self.contexts = [
            np.random.normal(size=(self.context_dim, ))
            for _ in range(self.n_contexts)
        ]
        self.reset()

    def reset(self):
        """reset the current context and hidden state to the 1st one

        Returns
        -------
        type
            Description of returned object.

        """
        self.current_context_id = 0
        self.h_prev = None

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_n_contexts(self, n_contexts):
        self.n_contexts = n_contexts

    def update(self, h_t):
        """given the current hidden state, update and return the context rep
        if
        - this is the 1st h_t, return the 0-th context rep
        else
        - update the current context if h_delta is big
        - then return the current context

        Parameters
        ----------
        h_t : 1d numpy array
            the current hidden state

        Returns
        -------
        ctx_t
            the current context representation

        """
        # if this is not the 1st hidden state
        if self.h_prev is not None:
            # calculate delta hidden state
            h_delta = np.linalg.norm(h_t - self.h_prev)
            # if delta h is big, change context
            if h_delta > self.threshold:
                self.current_context_id += 1
        # check for context id errors
        if self.current_context_id >= self.n_contexts:
            raise ValueError(f'#contexts exceeds the limit {self.n_contexts}')
        self.h_prev = h_t
        return self.contexts[self.current_context_id]

    # def __repr__(self):
    #     return


'''testing'''
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', palette='colorblind', context='talk')
    np.random.seed(0)

    def get_random_walk(dim=1, n_time_steps=50, loc=0, scale=1):
        x = np.zeros((n_time_steps, dim))
        x[0, :] = np.random.normal(size=dim)
        all_steps = np.random.normal(
            loc=loc, scale=scale, size=(n_time_steps, dim))
        return np.cumsum(all_steps, axis=0)

    # init
    n_time_steps = 200
    context_dim = 2
    input_dim = 1
    loc = 0
    scale = 1
    threshold = 2.5
    context_rep = ContextRep(context_dim=context_dim, threshold=threshold)
    x = get_random_walk(
        dim=input_dim, n_time_steps=n_time_steps, loc=loc, scale=scale
    )

    # compute context representation over time
    ctx = np.zeros((n_time_steps, context_dim))
    for t in range(n_time_steps):
        ctx[t] = context_rep.update(x[t, :])

    # compute empirical h delta
    h_delta = np.zeros(n_time_steps,)
    for t in np.arange(1, n_time_steps, 1):
        h_delta[t] = np.linalg.norm(x[t] - x[t - 1])

    # plot event boundaries over time
    f, axes = plt.subplots(3, 1, figsize=(12, 12))
    axes[0].plot(x)
    axes[0].set_ylabel('h')
    axes[1].plot(h_delta)
    axes[1].axhline(threshold, color='red', linestyle='--', alpha=.3)
    axes[1].set_ylabel('h delta')
    axes[2].plot(ctx, 'o')
    axes[2].set_ylabel('context rep (scalar)')
    axes[2].set_xlabel('time')
    # axes[1].plot(h_delta > threshold)
    event_bonds = np.where(h_delta > threshold)[0]
    for eb in event_bonds:
        axes[2].axvline(eb, color='red', linestyle='--', alpha=.3)
    sns.despine()
    # plt.imshow(ctx)
