from tensorlearn.decomposition.factorization import tt_factorization

import numpy as np

a=np.random.rand(10,5,4,10)
tt=tt_factorization(a,0.1)
print(tt.error)