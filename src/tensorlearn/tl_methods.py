
    

from tensorlearn.decomposition import tensor_train
from tensorlearn.decomposition import candecomp_parafac
from tensorlearn.decomposition import tucker
from tensorlearn.operations import tensor_operations as top
from tensorlearn.operations import matrix_operations as mop
from tensorlearn.tensor_completion import cp_completion


############################################################
############################################################
#####               decompositions
def auto_rank_tt(tensor,epsilon):
    return tensor_train.auto_rank_tt(tensor,epsilon)
    
def cp_als_rand_init(tensor, rank, iteration, random_seed=None):
    return candecomp_parafac.cp_als_rand_init(tensor, rank, iteration, random_seed)
    
def tucker_hosvd(tensor, epsilon):
    return tucker.tucker_hosvd(tensor, epsilon)
    
############################################################
############################################################
####               tensor operations

#### TT
def tt_to_tensor(factors):
    return top.tt_to_tensor(factors)
        
def tt_compression_ratio(factors):
    return top.tt_compression_ratio(factors)

def tt_tensor_shape(factors):
    return top.tt_tensor_shape(factors)

def tt_ranks(factors):
    return top.tt_ranks(factors)


    
    
## General
def tensor_resize(tensor, new_shape):
    return top.tensor_resize(tensor,new_shape)
    
def tensor_frobenius_norm(tensor):
    return top.tensor_frobenius_norm(tensor)
    
def unfold(tensor,n):
    return top.unfold(tensor,n)
    
def mode_n_product(tensor,matrix,n):
    return top.mode_n_product(tensor,matrix,n)
    

### CP

def cp_to_tensor(weights, factors):
    return top.cp_to_tensor(weights, factors)
    
def cp_compression_ratio(weights, factors):
    return top.cp_compression_ratio(weights,factors)
    
    
### Tucker

def tucker_to_tensor(core_factor,factor_matrices):
    return top.tucker_to_tensor(core_factor,factor_matrices)

def tucker_compression_ratio(core_factor,factor_matrices):
    return top.tucker_compression_ratio(core_factor,factor_matrices)

def tucker_tensor_shape(factor_matrices):
    return top.tucker_tensor_shape(factor_matrices)

def tucker_ranks(core_factor):
    return top.tucker_ranks(core_factor)



############################################################
############################################################
######              matrix operations

def error_truncated_svd(x, error):
    return mop.error_truncated_svd(x,error)

def column_wise_kronecker(a,b):
    return mop.column_wise_kronecker(a,b)


############################################################
############################################################
######              tensor completion

def cp_completion_als(tensor, samples, rank, iteration, cp_iteration=100):
    return cp_completion.cp_completion_als(tensor, samples, rank, iteration, cp_iteration=100)
    
    







