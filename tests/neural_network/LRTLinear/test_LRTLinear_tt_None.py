import torch.nn as nn
import torch
import tensorlearn as tl
import numpy as np

from tensorlearn.neural_network.torch import config, utils
from tensorlearn.neural_network.torch.low_rank_tensorized_layers import LRTLinear

config_dict_none = {
    'mode': None
}

None_config=config.LRTLinearConfig(**config_dict_none)



########### Linear From Torch #################
m=40
n=30
use_bias=True
W = torch.randn(n, m)
W_tensor=W.view(10,3,10,4)
W_tensor_array=W_tensor.numpy()

factors=tl.auto_rank_tt(W_tensor_array,0.01)

W_hat_tensor=tl.tt_to_tensor(factors)
W_hat_matrix=np.reshape(W_hat_tensor,(30,40)) 
W_hat_matrix_transpose=np.transpose(W_hat_matrix) #torch recieve W^T (output,input)


weight=torch.from_numpy(W_hat_matrix_transpose)
bias=torch.randn(40)
input_tensor=torch.randn(10,10,30)
factors_torch=[torch.from_numpy(f) for f in factors]

linear_layer = nn.Linear(in_features=n, out_features=m, bias=use_bias)
with torch.no_grad():
    linear_layer.weight.copy_(weight)
    if use_bias:
        linear_layer.bias.copy_(bias)

# Compute the output of the layer
output_linear = linear_layer(input_tensor)

#print(output_tensor.size())

ranks=tuple([f.shape[0] for f in factors]+[1])
dims=tuple(f.shape[1] for f in factors)

ttlinear_layer=LRTLinear('tt', dims,ranks,2, bias=use_bias)

with torch.no_grad():
    for i, f in enumerate(factors_torch):
        ttlinear_layer.factors[i].copy_(f)
    if use_bias:
        ttlinear_layer.bias.copy_(bias)

output_tt = ttlinear_layer(input_tensor)

print(output_linear[0,0,:])
print(output_tt[0,0,:])

error=output_linear-output_tt
print(tl.tensor_frobenius_norm(error.detach().numpy()))

