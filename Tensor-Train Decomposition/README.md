# Tensor-Train Decomposition

In the tensor train (TT) format, a $d$-way tensor $$\textbf{$\mathcal{Y}$} \in \mathbb{R}^{n_1\times .... \times n_d}$$ is approximated with a set of $d$ cores

$$\bar{\textbf{$\mathcal{G}$}}=\{\textbf{$\mathcal{G}$}_1, \textbf{$\mathcal{G}$}_2, ..., \textbf{$\mathcal{G}$}_d\}$$ 

where 

$$\textbf{$\mathcal{G}$}_j \in \mathbb{R}^{r_{j-1}\times n_j \times r_{j}}$$

, $r_j^'s$ for $j=1,...,d-1$ are the ranks, $r_0=r_d=1$, and each element of $\textbf{$\mathcal{Y}$}$ is approximated by :


$$E(\boldsymbol{\theta})=\frac{\lVert\textbf{$\mathcal{X}$}-\textbf{$\hat{\mathcal{X}}$}\lVert_F}{\lVert\textbf{$\mathcal{X}$}\lVert_F}$$
