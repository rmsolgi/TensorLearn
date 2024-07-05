import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils


class TTLinear(nn.Module):
    def __init__(self, dim_list, rank_list, tensor_input_order, bias=True):
        super(TTLinear, self).__init__()
        #self.out_features = out_features 
        self.rank_list = list(rank_list)
        self.dim_list = list(dim_list)
        self.tensor_input_order = tensor_input_order
        self.use_bias = bias
        #self.in_features=in_features
        self.dim = len(dim_list)
        #self.dot_prod = utils.tt_contract_x  # Placeholder for your low-rank tensor product class

        # Padding
        dim_array = np.array(dim_list)
        self.padded_num_in_features = np.prod(dim_array[:self.tensor_input_order])
        self.num_out_features=np.prod(dim_array[self.tensor_input_order:])

        # Initialize factors and bias
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(self.rank_list[i], self.dim_list[i], self.rank_list[i + 1]))
            for i in range(self.dim)
        ])
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(self.num_out_features))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, input_):
        #input_flatten = input_.view(-1, )
        num_in_dims = input_.dim()
        batch_shape=tuple(input_.shape[:-1])
        
        # Padding
        num_in_features = input_.size(-1)
        padding_size = self.padded_num_in_features - num_in_features

        if padding_size > 0:
            padding =tuple([0, padding_size]+[0] * (2 * (num_in_dims - 1)))
            input_padded = F.pad(input_, (0, padding))
        else:
            input_padded = input_

        # Reshape input to tensor format
        new_shape = batch_shape + tuple(self.dim_list[:self.tensor_input_order])
        
        reshaped_input_padded = input_padded.view(*new_shape)
        #tensor_shape = [-1] + 
        #input_tensor = input_padded.view(tensor_shape)

        # Tensor product
        tensor_product = utils.tt_contract_x(self.factors,reshaped_input_padded, num_in_dims-1)

        # Flatten tensor product and cut to the correct size
        new_tensor_shape=batch_shape+tuple([-1])
        tensor_product_reshaped = tensor_product.view(*new_tensor_shape)


        product = tensor_product_reshaped[...,0:self.num_out_features]

        if self.use_bias:
            output = product + self.bias
        else:
            output = product
        return output
    


class TuckerLinear(nn.Module):
    def __init__(self, dim_list, rank_list, tensor_input_order, bias=True):
        super(TuckerLinear, self).__init__()
        self.rank_list = list(rank_list)
        self.dim_list = list(dim_list)
        self.tensor_input_order = tensor_input_order
        self.use_bias = bias
        self.dim = len(dim_list)
        
        # Padding
        dim_array = np.array(dim_list)
        self.padded_num_in_features = np.prod(dim_array[:self.tensor_input_order])
        self.num_out_features = np.prod(dim_array[self.tensor_input_order:])

        # Initialize core factor and factor matrices
        self.core_factor = nn.Parameter(torch.randn(*self.rank_list))
        self.factor_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(self.dim_list[i], self.rank_list[i]))
            for i in range(self.dim)
        ])
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(self.num_out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        num_in_dims = input_.dim()
        batch_shape = tuple(input_.shape[:-1])
        
        # Padding
        num_in_features = input_.size(-1)
        padding_size = self.padded_num_in_features - num_in_features

        if padding_size > 0:
            padding = tuple([0, padding_size]+[0] * (2 * (num_in_dims - 1)))
            input_padded = F.pad(input_, padding)
        else:
            input_padded = input_

        # Reshape input to tensor format
        new_shape = batch_shape + tuple(self.dim_list[:self.tensor_input_order])
        reshaped_input_padded = input_padded.view(*new_shape)

        # Tensor product
        tensor_product = utils.tucker_contract_x([self.core_factor, self.factor_matrices], reshaped_input_padded, num_in_dims - 1)

        # Flatten tensor product and cut to the correct size
        new_tensor_shape=batch_shape+tuple([-1])
        tensor_product_reshaped = tensor_product.view(*new_tensor_shape)

        product = tensor_product_reshaped[..., :self.num_out_features]

        if self.use_bias:
            output = product + self.bias
        else:
            output = product
        
        return output