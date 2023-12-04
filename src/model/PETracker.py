import numpy as np

class PETracker():

    def __init__(self, t_max):
        self.t_max = t_max
        self.n_context = 0
        self.pe = {t: None for t in range(t_max)}
        self.tracker = {}
        self.add_context()

    def track_dict_for_one_context(self):
        return {t:[] for t in range(self.t_max)}

    def add_context(self):
        self.n_context +=1
        self.tracker[self.n_context] = self.track_dict_for_one_context()

    def record(self, context_id, t, pe, verbose=False):
        self.pe[t] = pe
        if not np.isnan(context_id):
            while context_id > self.n_context:
                self.add_context()
            self.validate_context_id_and_t(context_id, t)
            self.tracker[context_id][t].append(pe)
            if verbose:
                print(f'inferred context is {context_id}, recorded')
        else:
            if verbose:
                print('inferred context is np.nan, dont record')

    def get_mean(self, context_id, t):
        self.validate_context_id_and_t(context_id, t)
        if len(self.tracker[context_id][t]) == 0:
            return np.nan
        return np.mean(self.tracker[context_id][t])

    def recent_accuracy(self, t, n_trials=5):
        self.validate_t(t)
        if self.n_context == 0:
            return np.nan
        acc_t = []
        for c_id in range(self.n_context):
            acc_c_t = self.recent_accuracy_c_t(c_id, t)
            if acc_c_t is None:
                continue
        return np.mean(acc_t)

    def recent_accuracy_c_t(self, c_id, t, n_trials=5):
        if self.n_context == 0 or c_id < self.n_context:
            return None
        # if not enough trials, return None
        if len(self.tracker[c_id][t]) < n_trials:
            return None
        return np.mean(self.tracker[c_id][t][-n_trials:])

    def is_accurate_recently(self, t, threshold=1):
        if self.recent_accuracy(t) >= threshold:
            return True
        return False

    def validate_context_id_and_t(self, context_id, t):
        self.validate_t(t)
        if context_id not in self.tracker.keys():
            raise ValueError('context id doesnt exist')

    def validate_t(self, t):
        if t > self.t_max-1:
            raise ValueError('t too big')



if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    sns.set(style='white', palette='colorblind', context='poster')

    pet = PETracker(t_max = 7)
    print(pet.tracker)
    print()

    pet.record(1, 0, 2)
    print(pet.tracker)
    print()

    pet.record(2, 0, 2)
    print(pet.tracker)
    print()

    print(pet.get_mean(2,0))
