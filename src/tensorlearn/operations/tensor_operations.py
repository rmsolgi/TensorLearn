#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:00:44 2022

@author: Ryan Solgi
"""
import numpy as np
from tensorlearn.operations import tensor_operations, matrix_operations



      
    ###################### unfolding 
    ###################### ###################### ###################### ######################               
def unfold(tensor,n):
    
            return np.rollaxis(tensor, n, 0).reshape(tensor.shape[n], -1)
    
    ###################### determine the shape of tensor given tt factors
    
def mode_n_product(tensor,matrix,n):
    
    unfolded_tensor=np.swapaxes(tensor, n, 0).reshape(tensor.shape[n], -1)
    product=np.dot(matrix,unfolded_tensor)
    tensor_shape=np.array(tensor.shape)
    tensor_shape[0],tensor_shape[n]=tensor_shape[n],tensor_shape[0]

    tensor_shape[0]=product.shape[0]
    tensor_product=np.reshape(product,tensor_shape)
    
    return np.swapaxes(tensor_product,0,n)

    
    ###################### ###################### ###################### ######################
def tt_tensor_shape(factors):
    

    t_shape=tuple([f.shape[1] for f in factors])
    return t_shape

def tt_ranks(factors):
    ranks=tuple([f.shape[0] for f in factors]+[1])
    return ranks

    ###################### making tensor from factors 
    ###################### ###################### ###################### ######################  
def tt_to_tensor(factors): # factors is the list of facotrs

    
    t_shape=tensor_operations.tt_tensor_shape(factors)
    left_matrix=np.transpose(tensor_operations.unfold(factors[0],-1))
    for i in range (1,len(factors)):
            left_matrix=np.dot(left_matrix, tensor_operations.unfold(factors[i],0))
            left_matrix=np.reshape(left_matrix,(-1,factors[i].shape[-1]))
            
            
    tensor=np.reshape(left_matrix,t_shape)
    return tensor
    
    ###################### tensor resize, eneter zeros if the new size is bigger 
    ###################### ###################### ###################### ######################      
def tensor_resize(tensor,new_size): #new size is a tuple
    
    tensor_resized=tensor.copy()
    tensor_resized.resize(new_size)
    
    if tensor_resized.size < tensor.size:
        raise Exception ("new size is smaller than the origianl size") #does not allow data be missed
    
    else:
        return tensor_resized
    ###################### undo tensor resize using the factors
    ###################### ###################### ###################### ######################
    
def tt_to_tensor_undo_resize(factors, original_shape, original_size): #factors is the list of factors and the operation reverse tensor_resize function

    if np.prod(np.array(original_shape)) != original_size:
        raise Exception ("shape must match size")
        

    
    tensor=tensor_operations.tt_to_tensor(factors)
        
        
    if tensor.size < original_size:
        raise Exception ("tensor size is smaller than the requested size")
        
    tensor_flatten=tensor.flatten()
    tensor_cut=tensor_flatten[:original_size]
    
    tensor_undo_resize = np.reshape(tensor_cut,original_shape)
    
    return tensor_undo_resize

 
    ###################### TT Compression Ratio without reshaping
    ###################### ###################### ###################### ######################      
def tt_compression_ratio(factors):
    factors_size=0
    for item in factors:
        factors_size+=item.size
        
    t_shape=tensor_operations.tt_tensor_shape(factors)
    
    t_shape_array=np.array(t_shape)
    
    tensor_size=np.prod(t_shape_array)
    
    compression_ratio=tensor_size/factors_size
    
    return compression_ratio
        
    ###################### Tensor Frobenius norm
    ###################### ###################### ###################### ######################  
def tensor_frobenius_norm (tensor):
    tensor_flat=tensor.flatten()
    tensor_v=tensor_flat[...,np.newaxis]
    
    #tensor_v=np.reshape(tensor_flat,(1,(len(tensor_flat))))
    
    norm=np.linalg.norm(tensor_v,'fro')
    
    return norm


    ###################### CP tensor shape
    ###################### ###################### ###################### ###################


def cp_tensor_shape(factors):
    
    t_shape=[f.shape[0] for f in factors]
    
    return t_shape


    ###################### CP factors to tensor
    ###################### ###################### ###################### ###################
    
def cp_to_tensor(weights, factors):
    tensor_dim=len(factors)
    t_shape=tensor_operations.cp_tensor_shape(factors)
    rank=factors[0].shape[-1]
    rao_product=np.ones(shape=(1,rank))
    for i in range(1, tensor_dim):
                
                
        rao_product=matrix_operations.column_wise_kronecker(rao_product,factors[i])
    factors_0=weights*factors[0]
    tensor=np.dot(factors_0,np.transpose(rao_product))
        
    tensor=tensor.reshape(t_shape)
    return tensor
    

    ###################### CP Compression Ratio ccounting lambda (weights) too
    ###################### ###################### ###################### ######################      
def cp_compression_ratio(weights,factors):

    
    factors_size=weights.size
    for item in factors:
        factors_size+=item.size
        
    t_shape=tensor_operations.cp_tensor_shape(factors)
    
    t_shape_array=np.array(t_shape)
    
    tensor_size=np.prod(t_shape_array)
    
    compression_ratio=tensor_size/factors_size
    
    return compression_ratio    


    ###################### Tucker tensor shape
    ###################### ###################### ###################### ###################


def tucker_tensor_shape(factor_matrices):
    
    t_shape=tuple([f.shape[0] for f in factor_matrices])
    
    return t_shape

def tucker_ranks(core_factor):
    return core_factor.shape



    ###################### Tuckers factors to tensor
    ###################### ###################### ###################### ###################


def tucker_to_tensor(core_factor,factor_matrices):
    
    
    tensor=core_factor
    counter=0
    for factor in factor_matrices:
        tensor=tensor_operations.mode_n_product(tensor,factor,counter)
        counter+=1
    return tensor    
    
    ###################### Tucker Compression Ratio ccounting lambda (weights) too
    ###################### ###################### ###################### ######################      
def tucker_compression_ratio(core_factor,factor_matrices):

    
    factors_size=core_factor.size
    for item in factor_matrices:
        factors_size+=item.size
        
    t_shape=tensor_operations.tucker_tensor_shape(factor_matrices)
    
    t_shape_array=np.array(t_shape)
    
    tensor_size=np.prod(t_shape_array)
    
    compression_ratio=tensor_size/factors_size
    
    return compression_ratio   
                 
    ###################### tensor reshape
    ###################### ###################### ###################### ######################
        
def tensor_reshape(tensor, new_shape):
    return np.reshape(tensor,new_shape)
        
        
        
                
        
            
        
        
        

        
    
        
            
