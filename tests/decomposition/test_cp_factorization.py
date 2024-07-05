from tensorlearn.decomposition.factorization import cp_factorization

import numpy as np

a=np.random.rand(10,5,4,10)
decomp=cp_factorization(a,10,10)
print(decomp.factors)