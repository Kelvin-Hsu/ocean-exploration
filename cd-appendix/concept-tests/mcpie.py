import numpy as np

n_draws = 100
n_class = 4
n_query = 200

s = np.random.randint(0, n_class, (n_draws, n_query))
s[s == 2] == 0

# Entropy Computation
def H(p):
    p = p[p != 0]
    return - (p * np.log(p)).sum()

# Compute entropy separately for each query point
a = np.array([H(np.bincount(s[:, i])/n_draws) for i in np.arange(s.shape[1])])

def h(p):
    return - (p * np.log(p)).sum(0)

b = h(np.array([(s == c).mean(axis = 0) for c in np.arange(n_class)]))

print(a)
print(b)
print( a == b )