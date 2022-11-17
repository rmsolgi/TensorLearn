
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
[auto_rank_tt(tensor, error_bound)](#autoranktt-id)

### Tensor Operations for Tensor-Train 
[tt_to_tensor(factors)](#tttotensor-id)

[tt_compression_ratio(factors)](#ttcr-id)

### Tensor Operations
[tensor_resize(tensor,new_shape)](#tensorresize-id)

[unfold(tensor)](#unfold-id)

[tensor_frobenius_norm(tensor)](#tfronorm-id)

### Matrix Operations
[error_truncated_svd(x,error)](#etsvd-id)



## <a name="autoranktt-id"></a>auto_rank_tt

```python
tensorlearn.auto_rank_tt(tensor, epsilon)
```

This implementation of [tensor-train decomposition](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition) determines the ranks automatically based on a given error bound according to [Oseledets (2011)](https://epubs.siam.org/doi/10.1137/090752286). Therefore the user does not need to specify the ranks. Instead the user specifies an upper error bound (epsilon) which bounds the error of the decomposition. For more information and details please see the page [tensor-train decomposition](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition).

### Arguments 
@tensor < numpy array > - The given tensor to be decomposed.

@epsilon < float > - [The error bound of decomposition](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition#epsilon-id)

### Return
TT factors < list of numpy arrays > - The list includes numpy arrays of factors (or TT cores) according to TT decomposition. Length of the list equals the dimension of the given tensor to be decomposed.

[## Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)
## <a name="tttotensor-id"></a>tt_to_tensor

```python
tensorlearn.tt_to_tensor(factors)
```

Return the full tensor given the TT factors

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)

### Arguments
@factors < list of numpy arrays > - TT factors

### Return
full tensor < numpy array >

## <a name="ttcr-id"></a>tt_compression_ratio

```python
tensorlearn.tt_compression_ratio(factors)
```
Calculate [data compression ratio](https://en.wikipedia.org/wiki/Data_compression_ratio) for [tensor-train decompostion](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition)
### Arguments
@factors < list of numpy arrays > - TT factors

### Return
Compression ratio < float >









