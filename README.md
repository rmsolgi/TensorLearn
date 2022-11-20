
# TensorLearn

TensorLearn is a Python library distributed on [Pypi](https://pypi.org) ti implement 
tensor learning methods.

This is a package under development. Yet, the available methods are final and functional. The requirments is [Numpy](https://numpy.org).

    
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tensorlearn in Python.

```python
pip install tensorlearn
```

## methods
### Decomposition Methods
- [auto_rank_tt](#autoranktt-id)

- [cp_als_rand_init](#cpalsrandinit-id)

### Tensor Operations for Tensor-Train 
- [tt_to_tensor](#tttotensor-id)

- [tt_compression_ratio](#ttcr-id)

### Tensor Operations for CANDECOM/PARAFAC (CP)
- [cp_to_tensor](#cptotensor-id)

- [cp_compression_ratio](#cpcr-id)

### Tensor Operations
- [tensor_resize](#tensorresize-id)

- [unfold](#unfold-id)

- [tensor_frobenius_norm](#tfronorm-id)

### Matrix Operations
- [error_truncated_svd](#etsvd-id)

- [column_wise_kronecker](#colwisekron-id)

---


## <a name="autoranktt-id"></a>auto_rank_tt

```python
tensorlearn.auto_rank_tt(tensor, epsilon)
```

This implementation of [tensor-train decomposition](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition) determines the ranks automatically based on a given error bound according to [Oseledets (2011)](https://epubs.siam.org/doi/10.1137/090752286). Therefore the user does not need to specify the ranks. Instead the user specifies an upper error bound (epsilon) which bounds the error of the decomposition. For more information and details please see the page [tensor-train decomposition](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition).


### Arguments 
- tensor < array >: The given tensor to be decomposed.

- epsilon < float >: [The error bound of decomposition](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition#epsilon-id) in the range \[0,1\].

### Return
- TT factors < list of arrays >: The list includes numpy arrays of factors (or TT cores) according to TT decomposition. Length of the list equals the dimension of the given tensor to be decomposed.

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)

---
## <a name="cpalsrandinit-id"></a>cp_als_rand_init

```python
tensorlearn.cp_als_rand_init(tensor, rank, iteration, random_seed=None)
```

This is an implementation of [CANDECOMP/PARAFAC (CP) decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) using [alternating least squares (ALS) algorithm](https://arxiv.org/abs/2112.10855). 

### Arguments 
- tensor < array >: the given tensor to be decomposed

- rank < int >: number of ranks

- iterations < int >: the number of iterations of the ALS algorithm

- random_seed < int >: the seed of random number generator for random initialization of the factor matrices 


### Return
- weights < array >: the vector of normalization weights (lambda) in CP decomposition

- factors < list of arrays >: factor matrices of the CP decomposition
---

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/CP_decomposition/CP_example.py)



## <a name="tttotensor-id"></a>tt_to_tensor

```python
tensorlearn.tt_to_tensor(factors)
```

Returns the full tensor given the TT factors


### Arguments
- factors < list of numpy arrays >: TT factors

### Return
- full tensor < numpy array >

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)



---

## <a name="ttcr-id"></a>tt_compression_ratio

```python
tensorlearn.tt_compression_ratio(factors)
```

Returns [data compression ratio](https://en.wikipedia.org/wiki/Data_compression_ratio) for [tensor-train decompostion](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition)

### Arguments
- factors < list of numpy arrays >: TT factors

### Return
- Compression ratio < float >

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)

---

## <a name="cptotensor-id"></a>cp_to_tensor

Return the full tensor given the CP factor matrices and weights


```python
tensorlearn.cp_to_tensor(weights, factors)
```

### Arguments
- weights < array >: the vector of normalization weights (lambda) in CP decomposition

- factors < list of arrays >: factor matrices of the CP decomposition

### Return
- full tensor < array >

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/CP_decomposition/CP_example.py)


---


## <a name="cpcr-id"></a>cp_compression_ratio

Returns [data compression ratio](https://en.wikipedia.org/wiki/Data_compression_ratio) for [CP- decompostion](https://github.com/rmsolgi/TensorLearn/tree/Version-1.1.1/CP_decomposition)

```python
tensorlearn.cp_compression_ratio(weights, factors)
```

### Arguments
- weights < array >: the vector of normalization weights (lambda) in CP decomposition

- factors < list of arrays >: factor matrices of the CP decomposition

### Return

- Compression ratio < float >

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/CP_decomposition/CP_example.py)

---

## <a name="tensorresize-id"></a>tensor_resize

```python
tensorlearn.tensor_resize(tensor, new_shape)
```

This method reshapes the given tensor to a new shape. The new size must be bigger than or equal to the original shape. If the new shape results in a tensor of greater size (number of elements) the tensor fills with zeros. This works similar to [numpy.ndarray.resize()](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.resize.html)

### Arguments
- tensor < array >: the given tensor

- new_shape < tuple >: new shape 

### Return
- tensor < array >: tensor with new given shape

---

## <a name="unfold-id"></a>unfold
```python
tensorlearn.unfold(tensor, n)
```
Unfold the tensor with respect to dimension n.

### Arguments
- tensor < array >: tensor to be unfolded

- n < int >: dimension based on which the tensor is unfolded

### Return
- matrix < array >: unfolded tensor with respect to dimension n

---

## <a name="tfronorm-id"></a>tensor_frobenius_norm

```python
tensorlearn.tensor_frobenius_norm(tensor)
```

Calculates the [frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html) of the given tensor.

### Arguments
- tensor < array >: the given tensor

### Return
- frobenius norm < float >

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)

---


## <a name="etsvd-id"></a>error_truncated_svd

```python
tensorlearn.error_truncated_svd(x, error)
```
This method conducts a [compact svd](https://en.wikipedia.org/wiki/Singular_value_decomposition) and return [sigma (error)-truncated SVD](https://langvillea.people.cofc.edu/DISSECTION-LAB/Emmie%27sLSI-SVDModule/p5module.html) of a given matrix. This is an implementation using [numpy.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) with full_matrices=False. This method is used in [TT-SVD algorithm](https://github.com/rmsolgi/TensorLearn/tree/main/Tensor-Train%20Decomposition#ttsvd-id) in [auto_rank_tt](#autoranktt-id).

### Arguments
- x < 2D array >: the given matrix to be decomposed

- error < float >: the given error in the range \[0,1\]

### Return
- r, u, s, vh < int, numpy array, numpy array, numpy array > 


---

## <a name="colwisekron-id"></a>column_wise_kronecker

```python
tensorlearn.column_wise_kronecker(a, b)
```
Returns the column wise Kronecker product (Sometimes known as Khatri Rao) of two given matrices.

### Arguments

- a,b < 2D array >: the given matrices

### Return

- column wise Kronecker product < array >




