# Tensor-Train Decomposition

In the tensor train (TT) format, a $d$-way tensor $$\textbf{$\mathcal{Y}$} \in \mathbb{R}^{n_1\times .... \times n_d}$$ is approximated with a set of $d$ cores

$$\bar{\textbf{$\mathcal{G}$}}=\{\textbf{$\mathcal{G}$}_1, \textbf{$\mathcal{G}$}_2, ..., \textbf{$\mathcal{G}$}_d\}$$ 

where 

$$\textbf{$\mathcal{G}$}_j \in \mathbb{R}^{r_{j-1}\times n_j \times r_{j}}$$

, $r_js$ for $j=1,...,d-1$ are the ranks, $r_0=r_d=1$, and each element of $$\textbf{$\mathcal{Y}$}$$ is approximated by

$$\hat{\textbf{{$\mathcal{Y}$}}}[i_1,...,i_d]=\sum_{l_0,...,l_d} {\textbf{$\mathcal{G}$}_1[l_0,i_1,l_1]\textbf{$\mathcal{G}$}_2[l_1,i_2,l_2]...\textbf{$\mathcal{G}$}_{d}[l_{d-1},i_d,l_d]}$$


$$E(\boldsymbol{\theta})=\frac{\lVert\textbf{$\mathcal{X}$}-\textbf{$\hat{\mathcal{X}}$}\lVert_F}{\lVert\textbf{$\mathcal{X}$}\lVert_F}$$
