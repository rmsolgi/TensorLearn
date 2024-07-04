import numpy as np

import tensorlearn as tl

a=np.random.rand(10,10,10)
factors=tl.auto_rank_tt(a,0.01)

ranks=tl.tt_ranks(factors)
print(ranks)

shape=tl.tt_tensor_shape(factors)
print(shape)

core_factor,factor_matrices=tl.tucker_hosvd(a,0.01)

tucker_ranks=tl.tucker_ranks(core_factor)
tucker_shape=tl.tucker_tensor_shape(factor_matrices)

print(tucker_ranks)
print(tucker_shape)