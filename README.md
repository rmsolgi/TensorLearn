
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

@epsilon < float > - [The error bound of decomposition](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition#epsilon-id) in the range \[0,1\].

### Return
TT factors < list of numpy arrays > - The list includes numpy arrays of factors (or TT cores) according to TT decomposition. Length of the list equals the dimension of the given tensor to be decomposed.

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)

## <a name="tttotensor-id"></a>tt_to_tensor

```python
tensorlearn.tt_to_tensor(factors)
```

Return the full tensor given the TT factors


### Arguments
@factors < list of numpy arrays > - TT factors

### Return
full tensor < numpy array >

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)

## <a name="ttcr-id"></a>tt_compression_ratio

```python
tensorlearn.tt_compression_ratio(factors)
```
Calculate [data compression ratio](https://en.wikipedia.org/wiki/Data_compression_ratio) for [tensor-train decompostion](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition)

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)

### Arguments
@factors < list of numpy arrays > - TT factors

### Return
Compression ratio < float >

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)


## <a name="tensorresize-id"></a>tensor_resize

```python
tensorlearn.tensor_resize(tensor, new_shape)
```

Reshape the given tensor to a new shape. The new size must be bigger than or equal to the original shape. If the new shape results in a tensor of greater size (number of elements) the tensor fills with zeros. 

## Arguments
@tensor < numpy array > - the given tensor

@new_shape < tuple > - new shape 

## Return
tensor < numpy array > - tensor with new given shape


## <a name="unfold-id"></a>unfold
```python
tensorlearn.unfold(tensor, n)
```
Unfold the tensor with respect to dimension n.

## Arguments
@tensor < numpy array > - tensor to be unfolded

@n < int > - dimension based on which the tensor is unfolded

## Return
matrix < numpy array > - unfolded tensor with respect to dimension n


## <a name="tfronorm-id"></a>tensor_frobenius_norm

```python
tensorlearn.tensor_frobenius_norm(tensor)
```

Calculate the [frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html) of the given tensor.

## Arguments
@tensor < numpy array > - the given tensor

## Return
frobenius norm < float >

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)

---



## <a name="etsvd-id"></a>error_truncated_svd

```python
tensorlearn.error_truncated_svd(x, error)
```
Conduct a [compact svd](https://en.wikipedia.org/wiki/Singular_value_decomposition) and return [sigma (error)-truncated SVD](https://langvillea.people.cofc.edu/DISSECTION-LAB/Emmie%27sLSI-SVDModule/p5module.html) of a given matrix. This is an implementation using [numpy.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) with full_matrices=False. This method is used in [TT-SVD algorithm](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition#ttsvd-id) in [auto_rank_tt](#autoranktt-id).

## Arguments
@x < 2D numpy array > - the given matrix to be decomposed

@error < float > - the given error in the range \[0,1\]



