from scipy.special import kl_div
import numpy as np
def ber_kl_div(p, q):
    return kl_div(p,q) + kl_div(1-p,1-q)

def ber_kl_div(p, q):
    return p * (np.log(p/q)) + (1-p) * (np.log((1-p)/(1-q)))

small_num = .001
dim = 10

p_ = np.array([.5, .5]) # model
q_ = np.array([1, 0]) # target

p_ = np.array([.5, .5])
q_ = np.array([.3, .7])

# p_ = np.abs(np.random.normal(size=(dim)))
# p_ = np.eye(dim)[1]
# q_ = np.eye(dim)[0]

p_ = p_ + small_num
q_  = q_ +  small_num


def norm_by_sum(x):
    x = np.array(x)
    return x / x.sum()

p = norm_by_sum(p_)
q = norm_by_sum(q_)

kl = kl_div(p, q)
print(kl)
print(np.sum(kl))


[kl_div(p[i], q[i]) for i in range(len(p))]



# mykl = p * (np.log(p/q)) - p + q
# print(mykl)
