import numpy as np
import tensorlearn as tl

a=np.random.rand(100,10)
b=np.reshape(a,(10,10,10))

factors=tl.auto_rank_tt(b,0.5)
print(len(factors))