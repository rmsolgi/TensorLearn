

"""
@author: Ryan Solgi
"""
import numpy as np
import tensorlearn as tl

#lets build an arbitrary array and reshape it
tensor=np.arange(0,10000)
tensor=np.reshape(tensor,(20,50,10))

# CP decomposition, set rank 2 and iterations=50
weights, factors = tl.cp_als_rand_init(tensor, 2, 50)

# estimate tensor_hat using factors of decomposition
tensor_hat=tl.cp_to_tensor(weights, factors)

# Calculate the error between the estimated tensor using the factors and the original tensor
error=tensor_hat-tensor


error_ratio=tl.tensor_frobenius_norm(error)/tl.tensor_frobenius_norm(tensor)

print('error (%)= ',error_ratio*100)

# one application of tensor decomposition is data compression
# So, lets calculate the compression ratio
# Calculate data compression ratio for CP

cr = tl.cp_compression_ratio(weights, factors)

print('data compression ratio = ',cr)

# Calculate data saving 
ds= 1- (1/cr)

print('data_saving (%): ', ds*100)


