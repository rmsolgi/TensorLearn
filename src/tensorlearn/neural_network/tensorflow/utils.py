import tensorflow as tf

def tt_contract_x(factors,x, num_path_dims):

    num_dim_x=len(x.shape)-num_path_dims

    factor_0=tf.reshape(factors[0],shape=(factors[0].shape[1],factors[0].shape[2]))
    partial=tf.tensordot(x,factor_0,axes=[[num_path_dims],[0]])
        
    for i in range(1,num_dim_x):
            
        partial=tf.tensordot(partial,factors[i],axes=[[num_path_dims,-1],[1,0]])

    for i in range(num_dim_x,len(factors)-1):
        partial=tf.tensordot(partial,factors[i],axes=[[-1],[0]])
        
    factor_last=tf.reshape(factors[-1],shape=(factors[-1].shape[0],factors[-1].shape[1]))
    output=tf.tensordot(partial,factor_last,axes=[[-1],[0]])
        
    return output


def tucker_contract_x(factors,x, num_patch_dims):
    core_factor=factors[0]
    factor_matrices=factors[1]
    num_dim_x=len(x.shape)-num_patch_dims
    partial=tf.tensordot(x,factor_matrices[0],axes=[[num_patch_dims],[0]])

    for i in range(1,num_dim_x):
        partial=tf.tensordot(partial,factor_matrices[i],axes=[[num_patch_dims],[0]])
    axes=tf.range(num_dim_x)
    axes_p=axes+num_patch_dims
    axes=axes.numpy().tolist()
    axes_p=axes_p.numpy().tolist()
    output=tf.tensordot(partial,core_factor,axes=[axes_p,axes])
        
    for i in range (num_dim_x,len(factor_matrices)):
        output=tf.tensordot(output,factor_matrices[i],axes=[[num_patch_dims],[-1]])
    return output

