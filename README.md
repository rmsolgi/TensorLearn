
# tensorlearn

tensorlearn is a Python library distributed on [Pypi](https://pypi.org) for implementing 
tensor learning 

This is a package under development. Yet, the available methods are final and functional. The backend is [Numpy](https://numpy.org).

    
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tensorlearn in Python.

```python
pip install tensorlearn
```

## methods
### Decomposition Methods
[tensorlearn.auto_rank_tt(tensor, error_bound)](#autoranktt-id)

### Tensor Operations for Tensor-Train 
[tensorlearn.tt_to_tensor(factors)](#tttotensor-id)
[tensorlearn.tt_compression_ratio(factors)](#ttcr-id)

### Tensor Operations
[tensorlearn.tensor_resize(tensor,new_shape)](#tensorresize-id)
[tensorlearn.unfold(tensor)](#unfold-id)
[tensorlearn.tensor_frobenius_norm(tensor)](#tfronorm-id)

### Matrix Operations
[tensorlearn.error_truncated_svd(x,error)](#etsvd-id)



## <a name="autoranktt-id"></a>Tensor-Train Decomposition with Auto Rank Determination

tensorlearn.auto_rank_tt(tensor, epsilon)

This implementation of tensor-train decomposition determines rank automatically based on a given error bound written according to [Oseledets (2011)](https://epubs.siam.org/doi/10.1137/090752286). Therefore the user does not need to specify a rank. Instead the user specifies an upper error bound (epsilon) which is the [frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html) of the error divided by the frobenius norm of the given tensor to be decomposed.

### Arguments 
@tensor <numpy array> - The given tensor to be decomposed.

@epsilon <float> - Error bound = frobenius norm of the error / frobenius norm of the given tensor. 
### Outputs
@factors <list> - The list includes 2D numpy arrays of factors according to TT decomposition. Length of the list equals the dimension of the given tensor to be decomposed.





