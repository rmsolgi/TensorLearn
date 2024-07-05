

import copy
from tensorlearn.tensor_geometry import utils

#########################################################################
#########################################################################
class tss_node():
    def __init__(self,cardinality):
        self.cardinality=cardinality
        #self.ancestor_size=ancestor_size+1
        #self.data=None
        #self.rank=None
        #self.cost=None
        self.children=None #self.node_children_builder(ancestor_limit)

#########################################################################
#########################################################################
def tree_builder_helper(cardinality,ancestor_size, ancestor_limit):
    new_node=tss_node(cardinality)
    child_list=[]

    if cardinality>1: #otherwise children is None by default
        if ancestor_limit==None or ancestor_size<ancestor_limit:
            divisors=utils.divisors_finder(cardinality)
            for d in divisors:
                child_node=tree_builder_helper(d, ancestor_size+1,ancestor_limit)
                child_list.append(child_node)
        else:
            child_node=tss_node(1)
            child_list.append(child_node)
        new_node.children=child_list
    
    return new_node

#########################################################################
#########################################################################
# build the tree for divisors and possible cardinalities (dividend)

def tss_tree_builder(cardinality,ancestor_limit):
    tss_tree_root=tree_builder_helper(cardinality,1,ancestor_limit)
    return tss_tree_root

#########################################################################
#########################################################################
def count_nodes(root):
    queue=[]
    counter=1
    for c in root.children:
            queue.append(c)
            counter+=1
            

    while len(queue)!=0:
            node=queue.pop()
            if node.children != None:
                for d in node.children:
                    queue.append(d)
                    counter+=1
    return counter

#########################################################################
#########################################################################
def root_to_leaf_helper(node,path):
    global memory
    path.append(node.cardinality)

    if node.children!=None:
         for c in node.children:
             path=root_to_leaf_helper(c,path)
             path.pop()
    else:
        memory.append(copy.deepcopy(path))
    return path
        
#########################################################################
#########################################################################
# Find all toot to leaf pathes
def root_to_leaf(root):
    path=[]
    global memory
    memory=[]
    path=root_to_leaf_helper(root, path)
    return memory

#########################################################################
#########################################################################
#########################################################################
# get all leaves and output all possible shapes for dim>=3

def leaves_to_shapes(root):
    leaves=root_to_leaf(root)
    shapes_list=[]
    for leaf in leaves:
        dim_sizes=[]
        for i in range(1,len(leaf)):
            dim_sizes.append(int(leaf[i-1]/leaf[i]))
        if len(dim_sizes)>=2:
            shapes_list.append(dim_sizes)
    return shapes_list


def leaves_to_shapes_fixed_order(root,order):
    leaves=root_to_leaf(root)
    shapes_list=[]
    for leaf in leaves:
        dim_sizes=[]
        for i in range(1,len(leaf)):
            dim_sizes.append(int(leaf[i-1]/leaf[i]))
        if len(dim_sizes)>=2 and len(dim_sizes)==order:
            shapes_list.append(dim_sizes)
    return shapes_list

#########################################################################
#########################################################################


def get_tensor_shapes(cardinality, max_dim):
    root=tss_tree_builder(cardinality,max_dim)
    shapes_list=leaves_to_shapes(root)
    return shapes_list



def get_fixed_order_tensor_shapes(cardinality, order):
    root=tss_tree_builder(cardinality,order)
    shapes_list=leaves_to_shapes_fixed_order(root, order)
    return shapes_list

#########################################################################
#########################################################################

def dyadic_cartesian(cardinality1,cardinality2,order1,order2):
    collection=[]
    shapes_list_a=get_fixed_order_tensor_shapes(cardinality1, order1)
    shapes_list_b=get_fixed_order_tensor_shapes(cardinality2, order2)
    for i in shapes_list_a:
        for j in shapes_list_b:
            collection.append(i+j)
    
    return collection


