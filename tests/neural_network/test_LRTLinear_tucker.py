import sys
import os
import torch
import numpy as np
import tensorlearn as tl
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/tensorlearn/neural_network/torch')))
from low_rank_tensorized_layers import LRTLinear


########### Linear From Torch #################
m=40
n=30
use_bias=True
W = torch.randn(n, m)
W_tensor=W.view(10,3,10,4)
W_tensor_array=W_tensor.numpy()

core_factor,factor_matrices=tl.tucker_hosvd(W_tensor_array,0.01)

W_hat_tensor=tl.tucker_to_tensor(core_factor,factor_matrices)
W_hat_matrix=np.reshape(W_hat_tensor,(30,40)) 
W_hat_matrix_transpose=np.transpose(W_hat_matrix) #torch recieve W^T (output,input)


weight=torch.from_numpy(W_hat_matrix_transpose)
bias=torch.randn(40)
input_tensor=torch.randn(10,10,30)
factor_matrices_torch=[torch.from_numpy(f) for f in factor_matrices]
core_factor_torch=torch.from_numpy(core_factor)

linear_layer = nn.Linear(in_features=n, out_features=m, bias=use_bias)
with torch.no_grad():
    linear_layer.weight.copy_(weight)
    if use_bias:
        linear_layer.bias.copy_(bias)

# Compute the output of the layer
output_linear = linear_layer(input_tensor)

#print(output_tensor.size())

ranks=tuple([f.shape[1] for f in factor_matrices])
dims=tuple(f.shape[0] for f in factor_matrices)

tuckerlinear_layer=LRTLinear('tucker', dims,ranks,2, bias=use_bias)

with torch.no_grad():
    for i, f in enumerate(factor_matrices_torch):
        tuckerlinear_layer.factor_matrices[i].copy_(f)
    tuckerlinear_layer.core_factor.copy_(core_factor_torch)
    if use_bias:
        tuckerlinear_layer.bias.copy_(bias)

output_tt = tuckerlinear_layer(input_tensor)

print(output_linear[0,0,:])
print(output_tt[0,0,:])

error=output_linear-output_tt
print(tl.tensor_frobenius_norm(error.detach().numpy()))