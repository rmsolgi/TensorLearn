## CANDECOMP/PARAFAC (CP) Decomposition

CP decomposition is a tensor decomposition method. cp_als_rand_init is a method of TensorLearn package for implementation of CP decomposition using [ALS](#alsalgorithm-id) algorithm with random initialization of the [factor matrices](#factormatrices-id).


```python
tensorlearn.cp_als_rand_init(tensor,rank,iterations)
```

## Example
```python
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
```

According to the [CP decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) a tensor $X$ can be estimated by its factors as below:

$\hat{X} = \sum_r \lambda_{r} a_r^{(1)} \otimes a_r^{(2)} \otimes ... \otimes a_r^{(N)}$

where $a_r^{(n)} \in \mathbb{R}^{I_n}$ is a unit vector where $I_n$ is the size of the dimension $n$ with weight vector $\lambda \in \mathbb{R}^{R}$ where $R$ is the rank and $\otimes$ denotes outer product. 

![](https://github.com/rmsolgi/TensorLearn/blob/main/CP_decomposition/CP%20Fig.png)
[source (Minster et al., 2021)](https://arxiv.org/abs/2112.10855)

<a name="factormatrices-id"></a>Usually all $a_r^{(n)}$ for each mode $n$ for $n=1,2,...,N$ are collected as a matrix and called factor matrices, i.e., $A_n=[a_1^{(n)},a_2^{(n)},...,a_r^{(n)},...,a_R^{(n)}]$


<a name="alsalgorithm-id"></a>To compute CP decomposition, one method is alternative least squares (ALS) algorithm as below:

![](https://github.com/rmsolgi/TensorLearn/blob/main/CP_decomposition/cp_als_algorithm.png)
[source (Minster et al., 2021)](https://arxiv.org/abs/2112.10855)

