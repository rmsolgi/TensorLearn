import sys
import os
import torch
import numpy as np
import tensorlearn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/tensorlearn/neural_network/torch')))
import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/tensorlearn/decomposition')))
import tensor_train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

m, n, p = 30, 40, 50  


A = torch.randn(m, n)
B = torch.randn(n, p)

#C = torch.matmul(A, B)


x_tensor = A.reshape(3, 10, 10, 4)

W_tensor = B.reshape(10, 4, 10, 5)


tensor_array=W_tensor.numpy()

factors=tensor_train.auto_rank_tt(tensor_array,0.01)

W_hat_tensor=tensorlearn.tt_to_tensor(factors)

W_matrix=W_hat_tensor.reshape(40,50)

C=torch.matmul(A,torch.from_numpy(W_matrix))

torch_factors=[torch.from_numpy(s) for s in factors]

C_hat=utils.tt_contract_x(torch_factors,x_tensor,2)

C_tensor=C.reshape(3,10,10,5)

error=C_hat-C_tensor

frobenius_norm = torch.norm(error, p='fro')
print(frobenius_norm)

print(C_tensor[0,0,:,0])
print(C_hat[0,0,:,0])

