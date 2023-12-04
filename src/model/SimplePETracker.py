import numpy as np
from scipy.stats import sem


class SimplePETracker():

    def __init__(self, size=512):
        self.tracker = {}
        self.size = size

    def record(self, input, pe, verbose=False):
        input_key = str(input)
        if not self.input_in_tracker(input_key):
            self.tracker[input_key] = []
        self.tracker[input_key].append(pe)

        # if buffer is filled
        if len(self.tracker[input_key]) > self.size:
            self.tracker[input_key].pop(0)

    def get_recent_pe(self, input, n=1):
        input_key = str(input)
        if not self.input_in_tracker(input_key):
            return None
        return np.mean(self.tracker[input_key][-n:])

    def get_mean(self, input):
        input_key = str(input)
        if not self.input_in_tracker(input_key):
            return None
        return np.mean(self.tracker[input_key])

    def get_se(self, input):
        input_key = str(input)
        if not self.input_in_tracker(input_key):
            return None
        # return np.std(self.tracker[input_key]) / np.sqrt(len(self.tracker[input_key]))
        return sem(self.tracker[input_key])

    def get_std(self, input):
        input_key = str(input)
        if not self.input_in_tracker(input_key):
            return None
        return np.std(self.tracker[input_key])

    def input_in_tracker(self, input_key):
        return input_key in self.tracker.keys()

    def list_accuracy(self):
        return [v for k,v in self.tracker.items()]


if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    sns.set(style='white', palette='colorblind', context='poster')

    pet = SimplePETracker()
    print(pet.tracker)
    print()

    [x1, x2] = [
        [0,0,0],
        [1,1,1],
    ]
    data = [x1, x2, x1, x2]
    pe = [0, 1, 0, 1]

    for data_i, pe_i in zip(data, pe):
        pet.record(data_i, pe_i)
        print(pet.tracker)


    print()
