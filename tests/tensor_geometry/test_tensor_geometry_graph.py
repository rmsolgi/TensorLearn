import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/tensorlearn/tensor_geometry')))

import tensor_geometry_graph as tgg

tensor_size=100
max_tensor_order=3
tensor_shapes=tgg.get_tensor_shapes(tensor_size,3)
print(tensor_shapes)

tensor_shapes=tgg.get_fixed_order_tensor_shapes(tensor_size, 3)

print(tensor_shapes)

print(tgg.dyadic_cartesian(10,12,2,2))

