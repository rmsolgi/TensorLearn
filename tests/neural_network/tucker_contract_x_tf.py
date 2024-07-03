import sys
import os
import tensorflow as tf
import numpy as np
import tensorlearn

# Append the path to utils (assuming a similar path structure)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/tensorlearn/neural_network/tensorflow')))
import utils  # Make sure this imports correctly and is compatible with TensorFlow

# Set up the device
device = 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'
print(f'Using device: {device}')

# Define dimensions
m, n, p = 30, 40, 50  

# Create random tensors
A = tf.random.normal((m, n))
B = tf.random.normal((n, p))

# Reshape the tensors
x_tensor = tf.reshape(A, (3, 10, 10, 4))
W_tensor = tf.reshape(B, (10, 4, 10, 5))

# Convert tensor to numpy array for tensorlearn
tensor_array = W_tensor.numpy()

# Compute the Tucker decomposition
core_factor, factor_matrices = tensorlearn.tucker_hosvd(tensor_array, 0.01)

# Convert the Tucker decomposition back to a tensor
W_hat_tensor = tensorlearn.tucker_to_tensor(core_factor, factor_matrices)

# Reshape the tensor back to the original matrix form
W_matrix = tf.reshape(W_hat_tensor, (40, 50))

# Perform the matrix multiplication
C = tf.linalg.matmul(A, W_matrix)

# Convert the factor matrices and core factor to TensorFlow tensors
tf_factor_matrices = [tf.convert_to_tensor(s) for s in factor_matrices]
tf_core_factor = tf.convert_to_tensor(core_factor)

# Perform the tensor contraction
C_hat = utils.tucker_contract_x([tf_core_factor, tf_factor_matrices], x_tensor, 2)

# Reshape the result tensor
C_tensor = tf.reshape(C, (3, 10, 10, 5))

# Compute the error and Frobenius norm
error = C_hat - C_tensor
frobenius_norm = tensorlearn.tensor_frobenius_norm(error.numpy())
print(frobenius_norm)

# Print specific slices of the tensors
print(C_tensor[0, 0, :, 0])
print(C_hat[0, 0, :, 0])
