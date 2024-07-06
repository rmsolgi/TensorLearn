import tensorlearn.neural_network.torch.utils as tl
import torch
shape=tl.get_tensorized_layer_balanced_shape(100,50,2,2)
a=torch.rand(100,50)
b=a.view(shape)