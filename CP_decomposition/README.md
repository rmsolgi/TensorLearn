## CANDECOMP/PARAFAC (CP) Decomposition

CP decomposition is a tensor decomposition method. cp_als_rand_init is a method of TensorLearn package for implementing CP decomposition with random initialization of the [factor matrices](#factormatrices-id).


```python
tensorlearn.cp_als_rand_init(tensor,rank,iterations)
```


According to the CP decomposition a tensor $X$ can be estimated by its factors as below:

$\hat{X} = \sum_r \lambda_{r} a_r^{(1)} \otimes a_r^{(2)} \otimes ... \otimes a_r^{(N)}$

where $a_r^{(n)} \in \mathbb{R}^{I_n}$ is a unit vector where $I_n$ is the size of the dimension $n$ with weight vector $\lambda \in \mathbb{R}^{R}$ where $R$ is the rank and $\otimes$ denotes outer product. 

<a name="factormatrices-id"></a>Usually all $a_r^{(n)}$ for each mode $n$ for $n=1,2,...,N$ are collected as a matrix and called factor matrices, i.e., $A_n=[a_1^{(n)},a_2^{(n)},...,a_r^{(n)},...,a_R^{(n)}]$

![](https://github.com/rmsolgi/TensorLearn/blob/main/CP_decomposition/CP%20Fig.png)
[source (Minster et al., 2021)](https://arxiv.org/abs/2112.10855)

To compute CP decomposition, one method is alternative least squares (ALS) algorithm as below:

![](https://github.com/rmsolgi/TensorLearn/blob/main/CP_decomposition/cp_als_algorithm.png)
[source (Minster et al., 2021)](https://arxiv.org/abs/2112.10855)

