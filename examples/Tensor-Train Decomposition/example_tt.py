
"""
@author: Ryan Solgi
"""

import numpy as np
import tensorlearn as tl

#lets generate an arbitrary array 
tensor = np.arange(0,1000) 

#reshaping it into a higher (3) dimensional tensor

tensor = np.reshape(tensor,(10,20,5)) 
epsilon=0.05 
#decompose the tensor to its factors
tt_factors=tl.auto_rank_tt(tensor, epsilon) #epsilon is the error bound

#tt_factors is a list of three arrays which are the tt-cores

#rebuild (estimating) the tensor using the factors again as tensor_hat

tensor_hat=tl.tt_to_tensor(tt_factors)

#lets see the error

error_tensor=tensor-tensor_hat

error=tl.tensor_frobenius_norm(error_tensor)/tl.tensor_frobenius_norm(tensor)

print('error (%)= ',error*100) #which is less than epsilon
# one usage of tensor decomposition is data compression
# So, lets calculate the compression ratio
data_compression_ratio=tl.tt_compression_ratio(tt_factors)

#data saving
data_saving=1-(1/data_compression_ratio)

print('data_saving (%): ', data_saving*100)
