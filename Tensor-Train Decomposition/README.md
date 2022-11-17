#Tensor-Train Decomposition


Tensor-Train decomposition is a [tensor decomposition](https://en.wikipedia.org/wiki/Tensor_decomposition) method presented by [Oseledets (2011)](https://epubs.siam.org/doi/10.1137/090752286).

This implementation of tensor-train decomposition determines rank automatically based on a given error bound written according to TT-SVD Algorithm. Therefore the user does not need to specify ranks. Instead the user specifies an upper error bound [epsilon](#epsilon-id). 

In the tensor train (TT) format, a $d$-way [tensor](https://en.wikipedia.org/wiki/Tensor): $$\textbf{$\mathcal{Y}$} \in \mathbb{R}^{n_1\times .... \times n_d}$$ is approximated with a set of $d$ cores

$$\bar{\textbf{$\mathcal{G}$}}=\{\textbf{$\mathcal{G}$}_1, \textbf{$\mathcal{G}$}_2, ..., \textbf{$\mathcal{G}$}_d\}$$ 

where 

$$\textbf{$\mathcal{G}$}_j \in \mathbb{R}^{r_{j-1}\times n_j \times r_{j}}$$

, $r_js$ for $j=1,...,d-1$ are the ranks, $r_0=r_d=1$, and each element of the tensor $Y$ is approximated by

$$\hat{\textbf{{$\mathcal{Y}$}}}[i_1,...,i_d]=\sum_{l_0,...,l_d} {\textbf{$\mathcal{G}$}_1[l_0,i_1,l_1]\textbf{$\mathcal{G}$}_2[l_1,i_2,l_2]...\textbf{$\mathcal{G}$}_d[l_{d-1},i_d,l_d]}$$

Given an error bound (epsilon), the core factors, $g_js$, are computed using $d-1$ sequential [singular value decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) of the auxiliary matrices formed by [unfolding tensor]() $Y$ along different axes. This decomposition process is called the TT-SVD. The error bound refers to the [Frobenius norm]() of the error between estimated and original tensor divided by the the Frobenius norm of the original tensor as below:

<a> name="epsilon-id"></a>
$$epsilon\geq\frac{\lVert\textbf{$\mathcal{Y}$}-\textbf{$\hat{\mathcal{Y}}$}\lVert_F}{\lVert\textbf{$\mathcal{Y}$}\lVert_F}$$
