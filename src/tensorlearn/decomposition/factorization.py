from tensorlearn.decomposition import tensor_train as tt
from tensorlearn.decomposition import tucker
from tensorlearn.decomposition import candecomp_parafac as cp
from tensorlearn.operations import tensor_operations as top

class tt_factorization(object):
    def __init__(self,tensor,epsilon):
        self.tensor=tensor
        self.epsilon=epsilon
        
        self.factorize()

    def factorize(self):
        self.factors=tt.auto_rank_tt(self.tensor,self.epsilon)
        self.rank=top.tt_ranks(self.factors)
        self.compression_ratio=top.tt_compression_ratio(self.factors)
        self.error=top.tensor_frobenius_norm(top.tt_to_tensor(self.factors)-self.tensor)\
            /top.tensor_frobenius_norm(self.tensor)
        



class tucker_factorization(object):
    def __init__(self,tensor,epsilon):
        self.tensor=tensor
        self.epsilon=epsilon
        
        self.factorize()

    def factorize(self):
        self.core_factor, self.factor_matrices=tucker.tucker_hosvd(self.tensor,self.epsilon)
        self.rank=top.tucker_ranks(self.core_factor)
        self.compression_ratio=top.tucker_compression_ratio(self.core_factor,self.factor_matrices)
        self.error=top.tensor_frobenius_norm(top.tucker_to_tensor(self.core_factor,self.factor_matrices)-self.tensor)\
            /top.tensor_frobenius_norm(self.tensor)

class cp_factorization(object):
    def __init__(self,tensor, rank, iteration, random_seed=None):
        self.tensor=tensor
        self.rank=rank
        self.iteration=iteration
        self.random_seed=random_seed
        
        self.factorize()

    def factorize(self):
        self.weights, self.factors=cp.cp_als_rand_init(self.tensor,self.rank, self.iteration,self.random_seed)
        #self.rank=top.tucker_ranks(self.core_factor)
        self.compression_ratio=top.cp_compression_ratio(self.weights,self.factors)
        self.error=top.tensor_frobenius_norm(top.cp_to_tensor(self.weights,self.factors)-self.tensor)\
            /top.tensor_frobenius_norm(self.tensor)