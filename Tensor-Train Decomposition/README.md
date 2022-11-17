# Tensor-Train Decomposition


Tensor-Train decomposition is a [tensor decomposition](https://en.wikipedia.org/wiki/Tensor_decomposition) method presented by [Oseledets (2011)](https://epubs.siam.org/doi/10.1137/090752286).

```python
tensorlearn.auto_rank_tt(tensor, epsilon)
```

This implementation of tensor-train decomposition determines rank automatically based on a given error bound written according to TT-SVD Algorithm. Therefore the user does not need to specify ranks. Instead the user specifies an upper error bound [epsilon](#epsilon-id). 
```python

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
```
In the tensor train (TT) format, a $d$-way [tensor](https://en.wikipedia.org/wiki/Tensor): $$\textbf{$\mathcal{Y}$} \in \mathbb{R}^{n_1\times .... \times n_d}$$ is approximated with a set of $d$ cores

$$\bar{\textbf{$\mathcal{G}$}}=\{\textbf{$\mathcal{G}$}_1, \textbf{$\mathcal{G}$}_2, ..., \textbf{$\mathcal{G}$}_d\}$$ 

where 

$$\textbf{$\mathcal{G}$}_j \in \mathbb{R}^{r_{j-1}\times n_j \times r_{j}}$$

, $r_js$ for $j=1,...,d-1$ are the ranks, $r_0=r_d=1$, and each element of the tensor $Y$ is approximated by

$$\hat{\textbf{{$\mathcal{Y}$}}}[i_1,...,i_d]=\sum_{l_0,...,l_d} {\textbf{$\mathcal{G}$}_1[l_0,i_1,l_1]\textbf{$\mathcal{G}$}_2[l_1,i_2,l_2]...\textbf{$\mathcal{G}$}_d[l_{d-1},i_d,l_d]}$$

Given an error bound (<a name="epsilon-id"></a>epsilon), the core factors, $g_js$, are computed using $d-1$ sequential [singular value decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) of the auxiliary matrices formed by [unfolding tensor](https://github.com/rmsolgi/TensorLearn#unfold-id) $Y$ along different axes. This decomposition process is called the TT-SVD. The error bound refers to the [Frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html) of the error between estimated and original tensor divided by the the Frobenius norm of the original tensor as below:


$$epsilon\geq\frac{\lVert\textbf{$\mathcal{Y}$}-\textbf{$\hat{\mathcal{Y}}$}\lVert_F}{\lVert\textbf{$\mathcal{Y}$}\lVert_F}$$

## TT-SVD Algorithm

![](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/tt_svd_algorithm.png)
source: 
