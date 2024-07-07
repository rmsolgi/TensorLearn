import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tensorlearn.neural_network.torch import utils, config


class TTLinear(nn.Module):
    def __init__(self, dim_list, rank_list, tensor_input_order, bias=True):
        super(TTLinear, self).__init__()
        
        self.rank_list = list(rank_list)
        self.dim_list = list(dim_list)
        self.tensor_input_order = tensor_input_order
        self.use_bias = bias
       
        self.dim = len(dim_list)
        

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
        

        # Tensor product
        tensor_product = utils.tt_contract_x(self.factors,reshaped_input_padded, num_in_dims-1)

        # Flatten tensor product and cut to the correct size
        new_tensor_shape=batch_shape+tuple([-1])
        tensor_product_reshaped = tensor_product.view(*new_tensor_shape)


        product = tensor_product_reshaped[...,0:self.num_out_features]

        if self.use_bias:
            output = product + self.bias
            print('bias added')
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
        self.factors=nn.ParameterList([self.core_factor,self.factor_matrices])
        
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
        tensor_product = utils.tucker_contract_x(self.factors, reshaped_input_padded, num_in_dims - 1)

        # Flatten tensor product and cut to the correct size
        new_tensor_shape=batch_shape+tuple([-1])
        tensor_product_reshaped = tensor_product.view(*new_tensor_shape)

        product = tensor_product_reshaped[..., :self.num_out_features]

        if self.use_bias:
            output = product + self.bias
        else:
            output = product
        
        return output
    


class LRTLinear(nn.Module):
    def __init__(self, decomp_format, dim_list, rank_list, tensor_input_order, bias=True):
        super(LRTLinear, self).__init__()
        
        self.rank_list = list(rank_list)
        self.dim_list = list(dim_list)
        self.tensor_input_order = tensor_input_order
        self.use_bias = bias
       
        self.dim = len(dim_list)
        # Padding
        dim_array = np.array(dim_list)
        self.padded_num_in_features = np.prod(dim_array[:self.tensor_input_order])
        self.num_out_features=np.prod(dim_array[self.tensor_input_order:])
        self.format=decomp_format

        #initialize

        if decomp_format=='tt':
            self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(self.rank_list[i], self.dim_list[i], self.rank_list[i + 1]))
            for i in range(self.dim)])

            self.contract_func=utils.tt_contract_x
        elif decomp_format=='tucker':
            #self.core_factor = 
            #self.factor_matrices = nn.ParameterList([
            #nn.Parameter(torch.randn(self.dim_list[i], self.rank_list[i]))
            #for i in range(self.dim)])
            self.factors=nn.ParameterList([nn.Parameter(torch.randn(*self.rank_list)),nn.ParameterList([
            nn.Parameter(torch.randn(self.dim_list[i], self.rank_list[i]))
            for i in range(self.dim)])])

            self.contract_func=utils.tucker_contract_x
        else:
            raise TypeError("decomp_format is not supported, select 'tt' or 'tucker'")
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(self.num_out_features))
        else:
            self.register_parameter('bias', None)
    @classmethod
    def from_config(cls,config):
        if config.mode!='initial':
            raise ValueError('config mode must be initial.')
        decom_format=config.decomp_format
        dim_list=config.dim_list
        rank_list=config.rank_list
        tensor_input_order=config.in_order
        bias=config.bias
        instance=cls(decom_format,dim_list,rank_list,tensor_input_order,bias)
        return instance
        #if utils.check_none(format,dim_list,rank_list,tensor_input_order,bias):
        #    instance=cls(format,dim_list,rank_list,tensor_input_order,bias)
        #    return instance
        #else:
        #    raise ValueError('config includes None value for one or more vital parameters')

    @classmethod
    def from_teacher(cls,weight, bias, config):
        if config.mode!='teacher':
            raise ValueError('config mode must be teacher')
        
        if config.weight_transpose:
            weight=weight.t()
        if bias==None:
            use_bias=False
        else:
            use_bias=True

        #if config.in_order is not None:
        #    in_order=config.in_order
        #else:
        #    raise ValueError ("in_order is None, revise config")

        factors, rank_list, dim_list = utils.nn_tensor_geometry_optimization(weight,config)
        instance=cls(config.decomp_format, dim_list, rank_list, config.in_order, bias=use_bias)

        if config.decomp_format=='tt':
            instance.factors=nn.ParameterList([nn.Parameter(f) for f in factors]) 
        elif config.decomp_format=='tucker':
            instance.factors=nn.ParameterList([nn.Parameter(factors[0]),nn.ParameterList([
            nn.Parameter(f) for f in factors[1]])])
        else:
            raise ValueError("config.format is not recognized")

        if use_bias:
            instance.bias=nn.Parameter(bias)
        return instance
    
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
        tensor_product = self.contract_func(self.factors, reshaped_input_padded, num_in_dims - 1)

        # Flatten tensor product and cut to the correct size
        new_tensor_shape=batch_shape+tuple([-1])
        tensor_product_reshaped = tensor_product.view(*new_tensor_shape)

        product = tensor_product_reshaped[..., :self.num_out_features]

        if self.use_bias:
            output = product + self.bias
        else:
            output = product
        
        return output
