from tensorlearn.shape_search.ga_shape_search import ga
from tensorlearn.decomposition import tensor_train
from tensorlearn.decomposition import tucker
from tensorlearn.shape_search.random_shapes import random_search
import numpy as np

def shape_search_auto_rank_tt(tensor, epsilon, tensor_order, lower_bound_dim, reshape_func, algorithm_param, use_initial_shape=True):

    global tensor_size
    tensor_size=tensor.size

    global original_tensor
    original_tensor = tensor

    global error_bound
    error_bound = epsilon

    global reshape_function
    reshape_function=reshape_func

    lower_bound_dim=int(lower_bound_dim)

    tensor_order=int(tensor_order)

    tensor_shape=tensor.shape

    def function(x):
        
        
        new_tensor_shape=x.astype(int)
        
        one_dimensions=np.where(new_tensor_shape==1)
        new_tensor_shape=np.delete(new_tensor_shape,one_dimensions)

        reshaped_tensor=reshape_function(tensor, new_tensor_shape)

        factors_list=tensor_train.auto_rank_tt(reshaped_tensor, error_bound)
        factor_size=0
        for factor in factors_list:
            factor_size+=factor.size

        space_saving=1-(factor_size/tensor_size)

        cost=-space_saving

        return cost


    ga_model=ga(function=function,dimension=tensor_order,original_input_shape=tensor_shape,lower_bound=lower_bound_dim,initial_shape=use_initial_shape, algorithm_parameters=algorithm_param)
    ga_model.run()

    variable=ga_model.output_dict['variable']
    obj=ga_model.output_dict['function']

    one_dimensions=np.where(variable==1)
    variable=np.delete(variable,one_dimensions)

    solution={}
    solution['variable']=variable
    solution['space saving']=-obj
    
    report=np.array(ga_model.report)
    report=-report

    
    return solution, report





def shape_search_tucker_hosvd(tensor, epsilon, tensor_order, lower_bound_dim, reshape_func, algorithm_param, use_initial_shape=True):

    global tensor_size
    tensor_size=tensor.size

    global original_tensor
    original_tensor = tensor

    global error_bound
    error_bound = epsilon

    global reshape_function
    reshape_function=reshape_func

    lower_bound_dim=int(lower_bound_dim)

    tensor_order=int(tensor_order)

    tensor_shape=tensor.shape

    def function(x):
        
        
        new_tensor_shape=x.astype(int)
        
        one_dimensions=np.where(new_tensor_shape==1)
        new_tensor_shape=np.delete(new_tensor_shape,one_dimensions)

        reshaped_tensor=reshape_function(tensor, new_tensor_shape)

        core_factor,factor_matrices=tucker.tucker_hosvd(reshaped_tensor, error_bound)
        factor_size=core_factor.size
        for factor in factor_matrices:
            factor_size+=factor.size

        space_saving=1-(factor_size/tensor_size)

        cost=-space_saving

        return cost


    ga_model=ga(function=function,dimension=tensor_order,original_input_shape=tensor_shape,lower_bound=lower_bound_dim,initial_shape=use_initial_shape, algorithm_parameters=algorithm_param)
    ga_model.run()

    variable=ga_model.output_dict['variable']
    obj=ga_model.output_dict['function']

    one_dimensions=np.where(variable==1)
    variable=np.delete(variable,one_dimensions)

    solution={}
    solution['variable']=variable
    solution['space saving']=-obj
    
    report=np.array(ga_model.report)
    report=-report

    
    return solution, report



def random_shapes_auto_rank_tt(tensor, epsilon, tensor_order, lower_bound_dim, reshape_func, number_of_trials):

    global tensor_size
    tensor_size=tensor.size

    global original_tensor
    original_tensor = tensor

    global error_bound
    error_bound = epsilon

    global reshape_function
    reshape_function=reshape_func

    lower_bound_dim=int(lower_bound_dim)

    tensor_order=int(tensor_order)

    #tensor_shape=tensor.shape

    def function(x):
        
        
        new_tensor_shape=x.astype(int)
        
        one_dimensions=np.where(new_tensor_shape==1)
        new_tensor_shape=np.delete(new_tensor_shape,one_dimensions)

        reshaped_tensor=reshape_function(tensor, new_tensor_shape)

        factors_list=tensor_train.auto_rank_tt(reshaped_tensor, error_bound)
        factor_size=0
        for factor in factors_list:
            factor_size+=factor.size

        space_saving=1-(factor_size/tensor_size)

        cost=-space_saving

        return cost

    random_shape_model=random_search(function=function,input_tensor_size=tensor_size,dimension=tensor_order, low_bound_dim=lower_bound_dim,number_of_trials=number_of_trials)
    #ga_model=ga(function=function,dimension=tensor_order,original_input_shape=tensor_shape,lower_bound=lower_bound_dim,initial_shape=use_initial_shape, algorithm_parameters=algorithm_param)
    random_shape_model.run()

    variable=random_shape_model.best_shape
    obj=random_shape_model.best_function

    one_dimensions=np.where(variable==1)
    variable=np.delete(variable,one_dimensions)

    solution={}
    solution['variable']=variable
    solution['space saving']=-obj
    
    report=random_shape_model.results[:,-1]
    report=-report

    
    return solution, report



def random_shapes_tucker_hosvd(tensor, epsilon, tensor_order, lower_bound_dim, reshape_func, number_of_trials):

    global tensor_size
    tensor_size=tensor.size

    global original_tensor
    original_tensor = tensor

    global error_bound
    error_bound = epsilon

    global reshape_function
    reshape_function=reshape_func

    lower_bound_dim=int(lower_bound_dim)

    tensor_order=int(tensor_order)

    #tensor_shape=tensor.shape

    def function(x):
        
        
        new_tensor_shape=x.astype(int)
        
        one_dimensions=np.where(new_tensor_shape==1)
        new_tensor_shape=np.delete(new_tensor_shape,one_dimensions)

        reshaped_tensor=reshape_function(tensor, new_tensor_shape)

        core_factor,factor_matrices=tucker.tucker_hosvd(reshaped_tensor, error_bound)
        factor_size=core_factor.size
        for factor in factor_matrices:
            factor_size+=factor.size

        space_saving=1-(factor_size/tensor_size)

        cost=-space_saving

        return cost

    random_shape_model=random_search(function=function,input_tensor_size=tensor_size,dimension=tensor_order, low_bound_dim=lower_bound_dim,number_of_trials=number_of_trials)
    #ga_model=ga(function=function,dimension=tensor_order,original_input_shape=tensor_shape,lower_bound=lower_bound_dim,initial_shape=use_initial_shape, algorithm_parameters=algorithm_param)
    random_shape_model.run()

    variable=random_shape_model.best_shape
    obj=random_shape_model.best_function

    one_dimensions=np.where(variable==1)
    variable=np.delete(variable,one_dimensions)

    solution={}
    solution['variable']=variable
    solution['space saving']=-obj
    
    report=random_shape_model.results[:,-1]
    report=-report

    
    return solution, report
