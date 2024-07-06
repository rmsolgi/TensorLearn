import torch.nn as nn
import torch
import tensorlearn as tl
import numpy as np
from tensorlearn.neural_network.torch import config, utils
from tensorlearn.neural_network.torch.low_rank_tensorized_layers import LRTLinear

m=40
n=30
use_bias=True
W = torch.randn(n, m)
W_tensor=W.view(10,3,10,4)
if not use_bias:
    bias=None
else:
    bias=torch.randn(40)


config_dict_teacher = {
    'mode': 'teacher',
    'decomp_format': 'tt',
    'weight_transpose': True,
    'in_order': 2,
    'out_order':2,
    'error':1,
    'shape_search_method':'balanced'
}

teacher_config=config.LRTLinearConfig(**config_dict_teacher)
input_tensor=torch.randn(10,10,30)

ttlinear_layer_from_teacher=LRTLinear.from_teacher(W.t(),bias,teacher_config) #W.t() given torch layer has W^T
output_tt_from_teacher = ttlinear_layer_from_teacher(input_tensor)
print(output_tt_from_teacher[0,0,:])

with torch.no_grad():
    weight_factors=ttlinear_layer_from_teacher.factors
    weight_factors_array=[f.numpy() for f in weight_factors]
#print('outside', weight_factors_array[0])
weight_2=tl.tt_to_tensor(weight_factors_array)
weight_matrix_reshape=np.reshape(weight_2,(30,40)) 
weight_matrix_transpose=np.transpose(weight_matrix_reshape)

weight_2=torch.from_numpy(weight_matrix_transpose)

bias_2=ttlinear_layer_from_teacher.bias
print('bias', bias_2)
linear_layer_2 = nn.Linear(in_features=n, out_features=m, bias=use_bias)
with torch.no_grad():
    linear_layer_2.weight.copy_(weight_2)
    if use_bias:
        linear_layer_2.bias.copy_(bias_2)

output_tt_2 = linear_layer_2(input_tensor)

print(output_tt_2[0,0,:])