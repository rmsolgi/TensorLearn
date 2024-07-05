from tensorlearn.tensor_geometry import tensor_geometry_graph as tgg


def get_balanced_shape(num_in_features, num_out_features, in_order, out_order):
    shapes_list=tgg.dyadic_cartesian(num_in_features, num_out_features, in_order, out_order)
    optimal_shape = min(shapes_list, key=sum)
    return optimal_shape




    
