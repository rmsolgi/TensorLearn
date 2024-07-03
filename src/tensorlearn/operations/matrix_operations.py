#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 18:52:27 2022

@author: Ryan Solgi
"""


import numpy as np




def error_truncated_svd(x,threshold): #calcualte the r accroding to threshold and output the left and right matrices based on the rank
            
            shape_x=x.shape
            
            r_max=np.min(shape_x)
    
            u, singulars, vh = np.linalg.svd(x,full_matrices=False)
    
            s=np.diag(singulars)
            us=np.dot(u,s)
            
            
            r=0
    
            #sigma=np.linalg.norm(x,'fro')
            sigma=(np.sum(singulars**2))**(0.5)
            while sigma>threshold and r<r_max:
                
                r+=1
                
                
                singular_error=0
                for j in range(r,r_max):
                    singular_error+=singulars[j]**2
                    
                singular_error=singular_error**(0.5)
     
                
                
               
            
                #x_test=np.zeros(shape=shape_x)
                
                
                #x_test=np.dot(us[:,:r],vh[:r,:])
    
              
                
                #error=x-x_test
                
                #norm_error=np.linalg.norm(error,'fro')
                
                #sigma=norm_error#/np.linalg.norm(x,'fro')
                sigma=singular_error
                
                
            #for j in range(0,r):
            
            return r, u, s, vh
        

####################################################################################
####################################################################################
####################################################################################

 ########################### Khatri_rao product
 
def column_wise_kronecker(a,b): #also known as Khatri Rao product
 
        axis=0
        def kron(vector,split_point):
        
            return np.kron(vector[:split_point],vector[split_point:])
        
        array=np.concatenate((a,b),axis)
        
        kr_prod=np.apply_along_axis(kron,axis,array,split_point=a.shape[axis])

        return kr_prod
        

