from tensorlearn.decomposition.factorization import tucker_factorization

import numpy as np

a=np.random.rand(10,5,4,10)
tucker=tucker_factorization(a,1)
print(tucker.compression_ratio)