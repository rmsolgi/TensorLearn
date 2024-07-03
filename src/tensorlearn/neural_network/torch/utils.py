import torch


def tt_contract_x(factors,x,num_batch_dims): #for fully connected layer num_batch_dims=1, for CNN is 3

    num_dim_x = len(x.shape) - num_batch_dims #skip batch dims

    factor_0 = factors[0].reshape(factors[0].shape[1], factors[0].shape[2])
    partial = torch.tensordot(x, factor_0, dims=([num_batch_dims], [0]))
    
    for i in range(1, num_dim_x):
        partial = torch.tensordot(partial, factors[i], dims=([num_batch_dims, -1], [1, 0]))

    for i in range(num_dim_x, len(factors) - 1):
        partial = torch.tensordot(partial, factors[i], dims=([-1], [0]))
    
    factor_last = factors[-1].reshape(factors[-1].shape[0], factors[-1].shape[1])
    output = torch.tensordot(partial, factor_last, dims=([-1], [0]))
    
    return output


def tucker_contract_x(factors, x, num_batch_dims): #for fully connected layer num_batch_dims=1, for CNN is 3
    core_factor = factors[0]
    factor_matrices = factors[1]
    num_dim_x = len(x.shape) - num_batch_dims
    
    partial = torch.tensordot(x, factor_matrices[0], dims=([num_batch_dims], [0]))

    for i in range(1, num_dim_x):
        partial = torch.tensordot(partial, factor_matrices[i], dims=([num_batch_dims], [0]))
    
    axes = torch.arange(num_dim_x)
    axes_p = axes + num_batch_dims
    axes = axes.tolist()
    axes_p = axes_p.tolist()
    
    output = torch.tensordot(partial, core_factor, dims=(axes_p, axes))
    
    for i in range(num_dim_x, len(factor_matrices)):
        output = torch.tensordot(output, factor_matrices[i], dims=([num_batch_dims], [-1]))
    
    return output


    