'''
let the neural network loop over different context to compute likelihood
use the contextual bandit task
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from task import ContextualBandit
from model import CRPLSTM
sns.set(style='white', palette='colorblind', context='talk')
np.random.seed(1)
