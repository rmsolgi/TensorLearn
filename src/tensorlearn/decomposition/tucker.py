#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:20:18 2023

@author: ryan
"""
from tensorlearn.operations import tensor_operations as top
from tensorlearn.operations import matrix_operations as mop

import numpy as np

def tucker_hosvd(tensor, epsilon):
    
    
    t_dim=len(tensor.shape)
    
    
    factor_matrices=[]
    factor_list=np.arange(0,t_dim)
    
    sigma=(epsilon*top.tensor_frobenius_norm(tensor))/(t_dim**(0.5))
    
    # initialization with HOSVD
    for i in range(0,t_dim):
        unfolded_tensor=top.unfold(tensor,i)
        
        r,u,s,vh=mop.error_truncated_svd(unfolded_tensor,sigma)
        
        factor_matrix=u[:,:r]
        
        factor_matrices.append(factor_matrix)
        
    core_factor=tensor
    for i in factor_list:
        core_factor=top.mode_n_product(core_factor,np.transpose(factor_matrices[i]),i)
        

    return core_factor, factor_matrices





