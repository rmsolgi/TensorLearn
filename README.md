# TensorLearn


This is a library for some tensor decomposition and tensor-based methods. This is a project under development and more methods will be added. The current methods are functional.
    
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tensorlearn in Python.

```python
pip install tensorlearn
```

## methods


### Tensor Train Decomposition

- [auto_rank_tt](#autoranktt-id)
- [tt_to_tensor](#tttotensor-id)
- [tt_compression_ratio](#ttcr-id)

### CANDECOMP/PARAFAC (CP) Decomposition
- [cp_als_rand_init](#cpalsrandinit-id)
- [cp_to_tensor](#cptotensor-id)
- [cp_compression_ratio](#cpcr-id)

### Tucker Decomposition 
- [tucker_hosvd](#tuckerhosvd-id)
- [tucker_to_tensor](#tuckertotensor-id)
- [tucker_compression_ratio](#tuckercr-id)
### Tensor Completion using CP and ALS
- [cp_completion_als](#cpcompletionals-id)


### Tensor Operations
- [tensor_resize](#tensorresize-id)

- [unfold](#unfold-id)

- [tensor_frobenius_norm](#tfronorm-id)

### Matrix Operations
- [error_truncated_svd](#etsvd-id)

- [column_wise_kronecker](#colwisekron-id)

---

## <a name="cpcompletionals-id"></a>cp_completion_als
```python
tensorlearn.cp_completion_als(tensor, samples, rank, iteration, cp_iteration=100)
```
This implementation for tensor completion is based on CP decomposition given a fixed rank.

### Arguments 
- tensor < array >: The given tensor to be decomposed.
- samples < array >:  An array of `0`s and `1`s where `1`s represent observed samples, and `0`s indicate missing entries. This array's size must match the dimensions of the tensor.
- rank < int >: The rank for CP decomposition.
- iteration < int >: The iteration for the ALS algorithm.
- cp_iteration < int >: The iteration for initialization.

### Return
- weights < array >: the vector of normalization weights (lambda) in CP decomposition

- factors < list of arrays >: factor matrices of CP decomposition

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

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/examples/Tensor-Train%20Decomposition/example_tt.py)

---
## <a name="cpalsrandinit-id"></a>cp_als_rand_init

```python
tensorlearn.cp_als_rand_init(tensor, rank, iteration, random_seed=None)
```

This is an implementation of [CANDECOMP/PARAFAC (CP) decomposition](https://github.com/rmsolgi/TensorLearn/tree/main/CP_decomposition) using [alternating least squares (ALS) algorithm](https://arxiv.org/abs/2112.10855) with random initialization of factors.

### Arguments 
- tensor < array >: the given tensor to be decomposed

- rank < int >: number of ranks

- iterations < int >: the number of iterations of the ALS algorithm

- random_seed < int >: the seed of random number generator for random initialization of the factor matrices 


### Return
- weights < array >: the vector of normalization weights (lambda) in CP decomposition

- factors < list of arrays >: factor matrices of CP decomposition

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/examples/CP_decomposition/CP_example.py)

---

## <a name="tuckerhosvd-id"></a>tucker_hosvd
```python
tensorlearn.tucker_hosvd(tensor, epsilon)
```
### Arguments 
- tensor < array >: the given tensor to be decomposed
- epsilon < float >: The error bound of decomposition in the range \[0,1\].

### Return
- core_factor < array >: Core tensor factor of Tucker 
- factor_matrices < list >: A list of factor matrices of Tucker

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

Returns [data compression ratio](https://en.wikipedia.org/wiki/Data_compression_ratio) for [tensor-train decompostion](https://github.com/rmsolgi/TensorLearn/tree/main/CP_decomposition)

### Arguments
- factors < list of numpy arrays >: TT factors

### Return
- Compression ratio < float >

[Example](https://github.com/rmsolgi/TensorLearn/blob/main/Tensor-Train%20Decomposition/example_tt.py)

---

## <a name="cptotensor-id"></a>cp_to_tensor

Returns the full tensor given the CP factor matrices and weights


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


## <a name="tuckertotensor-id"></a>tucker_to_tensor

Returns the full tensor given the Tucker core factor and factor matrices


```python
tensorlearn.tucker_to_tensor(core_factor,factor_matrices)
```

### Arguments
- core_factor < array >: Core factor of Tucker decomposition

- factor_matrices < list of arrays >: factor matrices of Tucker decomposition

### Return
- full tensor < array >
---


## <a name="cpcr-id"></a>cp_compression_ratio

Returns [data compression ratio](https://en.wikipedia.org/wiki/Data_compression_ratio) for [CP- decompostion](https://github.com/rmsolgi/TensorLearn/tree/main/CP_decomposition)

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


## <a name="tuckercr-id"></a>Tucker_compression_ratio

Returns data compression ratio for Tucker decomposition.

```python
tensorlearn.tucker_compression_ratio(core_factor,factor_matrices)
```
### Arguments
- core_factor < array >: Core factor of Tucker decomposition

- factor_matrices < list of arrays >: factor matrices of Tucker decomposition

### Return

- Compression ratio < float >
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
Returns the [column wise Kronecker product (Sometimes known as Khatri Rao)](https://en.wikipedia.org/wiki/Khatri–Rao_product) of two given matrices.

### Arguments

- a,b < 2D array >: the given matrices

### Return

- column wise Kronecker product < array >

---
