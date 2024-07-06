import sys
import os
import torch
import numpy as np
import tensorlearn as tl
import torch.nn as nn


from tensorlearn.neural_network.torch.low_rank_tensorized_layers import TTLinear

########### Linear From Torch #################
m=40
n=30
use_bias=False
W = torch.randn(n, m)
W_tensor=W.view(10,3,10,4)
W_tensor_array=W_tensor.numpy()

factors=tl.auto_rank_tt(W_tensor_array,0.01)

W_hat_tensor=tl.tt_to_tensor(factors)
W_hat_matrix=np.reshape(W_hat_tensor,(30,40)) 
W_hat_matrix_transpose=np.transpose(W_hat_matrix) #torch recieve W^T (output,input)


