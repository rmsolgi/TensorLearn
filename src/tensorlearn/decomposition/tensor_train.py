#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 15:49:47 2022

@author: Ryan Solgi
"""


from tensorlearn.operations import tensor_operations as top
from tensorlearn.operations import matrix_operations as mop
#from tensorlearn.decomposition import tensor_train
import numpy as np


    
def auto_rank_tt(tensor,epsilon): #epsilon = error bound [0,1]
        
        
        #if tensor.ndim>=3 and epsilon >=0:

            t_shape=tensor.shape
            t_dim=len(tensor.shape)
            matrix=top.unfold(tensor,t_dim-1)
            
            sigma=epsilon*np.linalg.norm(matrix,'fro')/((t_dim-1)**0.5)
            
            modified_rank_list=[1]
            counter=1
            tt_list=[]
            r_list=[]
            
            while counter<t_dim:
                
                        r,u,s,vh=mop.error_truncated_svd(matrix,sigma)
                        
                        left_matrix=u[:,:r]
                        right_matrix=np.dot(s[:r,:r],vh[:r,:])
                        
                        
                        
                        
                        tt_list.append(left_matrix)
                        r_list.append(r)
                        m_shape=right_matrix.shape
                        new_shape=[int(r*t_shape[t_dim-1-counter]),int(m_shape[1]/t_shape[t_dim-1-counter])]
                        matrix=np.reshape(right_matrix,new_shape,order='F')
                        #matrix=mp.matrix_reshape(right_matrix,new_shape)
                        counter+=1
                        modified_rank_list.append(r)
                        
            modified_rank_list.append(1)
            tt_list.append(right_matrix)
            
            factors_list=[]
            
            for i in range(0,len(tt_list)):
                
                        transpose_core=np.transpose(tt_list[-i-1])
                        core_reshape=np.reshape(transpose_core,(modified_rank_list[-1-i],t_shape[i],modified_rank_list[-1-(i+1)]))
                        factors_list.append(core_reshape)
                    
            return factors_list
        
        #else:
            #raise Exception('tensor must be at leaset of dimension 3')
'''
################################################################################
################################################################################
#################################         TT decomposition with reshaping
def reshaping_tt(tensor,new_shape,epsilon):
    
    original_shape=tensor.shape
    
    tensor_resized=top.tensor_resize(tensor, new_shape)
    
    factors=tensor_train.auto_rank_tt(tensor_resized,epsilon)
    
    return(factors,original_shape)

################################################################################
'''

    
    
    
    
    
    
    
    
    
            
         
            
    
