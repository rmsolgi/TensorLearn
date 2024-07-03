#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 15:49:47 2022

@author: Ryan Solgi
"""


from tensorlearn.operations import tensor_operations as top
from tensorlearn.operations import matrix_operations as mop
import numpy as np


################################################# cp factorization using
#Alternating Least Squares (ALS) algorithm and random initialization

def cp_als_rand_init(tensor, rank, iteration, random_seed=None):
    
    tensor_dim=np.ndim(tensor)
    factors=[]
    
    for i in range(tensor_dim):
        
        np.random.seed(random_seed)
        factors.append(np.random.rand(tensor.shape[i],rank))
        
    for t in range(0,iteration):
        
        for n in range(tensor_dim):
            v=np.ones(shape=(rank,rank)) #v is updated as elementwise product of factors
            for i in range(tensor_dim):
                if i!=n:
                    product=np.dot(np.transpose(factors[i]),factors[i])
                    v=np.multiply(product,v)
                    
            rao_product=np.ones(shape=(1,rank)) #rao_proudct will be updated
            
            for i in range(tensor_dim):
                
                if i!=n:
                    rao_product=mop.column_wise_kronecker(rao_product,factors[i])
          
            tensor_n=top.unfold(tensor,n)
            
            tensor_rao_product=np.dot(tensor_n,rao_product)
    
            inverse=np.linalg.pinv(v)
            
            factors[n]=np.dot(tensor_rao_product,inverse)
            
            weights=np.linalg.norm(factors[n],axis=0)
            
            factors[n]=factors[n]/weights
    
    
    
    
    return weights, factors


####################################################################################
####################################################################################
####################################################################################
    
    
   
    
    
    
    
    
    
            
         
            
    
