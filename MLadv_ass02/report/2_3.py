import numpy as np
from Tree import Tree
from Tree import Node

from collections import defaultdict

def hej_child(node, topology, beta):
    children = []
    for index, parent in enumerate(topology):
        if parent == node:
            children.append(index)
    return children

def CPD(theta, node, cat, parent_cat=None):
    if parent_cat == None:
        return theta[node][cat]
    else:
        return theta[node][int(parent_cat)][int(cat)]

def s_root(tree_topology, theta, beta):
    prob = 0
    s_dict = defaultdict(dict)
    
    def S(u, j, children):
        if s_dict[u].get(j) != None:
            return s_dict[u][j]
        # One child
        if len(children) < 1:
            if beta.astype(int)[u] == j:
                s_dict[u][j] = 1
                return 1
            else:
                s_dict[u][j] = 0
                return 0
        # More than one child
        result = np.zeros(len(children))
        for child_nr, child in enumerate(children):
            for category in range(len(theta[0])):
                result[child_nr] += S(child, category, hej_child(child, tree_topology, beta)) * CPD(theta, child, category, j)
        s_result = np.prod(result)
        s_dict[u][j] = s_result
        return s_result
    # get s values in the root
    for i, th in enumerate(theta[0]):
        prob += S(0, i, hej_child(0, tree_topology, beta)) * CPD(theta, 0, i)
    return s_dict

# function to find their siblings
def hej_sibling(u, topology):
    for node, parent in enumerate(topology):
        if np.isnan(parent) and np.isnan(topology[u]) and u != node:
            return node
        elif parent == topology[u] and u != node:
            return node
    # AFTER searching all nodes, but there is no siblings -> return None
    return None

def calculate_likelihood(tree_topology, theta, beta):
    """
        This function calculates the likelihood of a sample of leaves.
        :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
        :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
        :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
        Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
        :return: likelihood: The likelihood of beta. Type: float.
        
        You can change the function signature and add new parameters. Add them as parameters with some default values.
        i.e.
        Function template: def calculate_likelihood(tree_topology, theta, beta):
        You can change it to: def calculate_likelihood(tree_topology, theta, beta, new_param_1=[], new_param_2=123):
        """
    
    # TODO Add your code here
    
    # preparation
    s_dict = s_root(tree_topology, theta, beta)
    t_dict = defaultdict(dict)
    
    # function forb calculating t dynamicakly and get the result
    def t_dyn(u, i, parent, sibling):
        if t_dict[u].get(i) != None:
            return t_dict[u][i]
        if np.isnan(parent):
            return CPD(theta, u, i) * s_dict[u][i]
        if sibling == None:
            result = 0
            for j in range(len(theta[0])):
                result += CPD(theta, u, i, j) * t_dyn(int_parent, j, tree_topology[int_parent], hej_sibling(int_parent, tree_topology))
                t_dict[u][i] = result
            return result
        # Siblings also should be taken into account in this case
        int_parent = int(parent)
        result = 0
        for j in range(len(theta[0])):
            for k in range(len(theta[0])):
                result += CPD(theta,u,i,j) * CPD(theta,sibling,k,j) * s_dict[sibling][k] * t_dyn(int_parent,j,tree_topology[int_parent],hej_sibling(int_parent,tree_topology))
        t_dict[u][i] = result
        return result
    
    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    
    print("Calculating the likelihood...")
    likelihood = 1.0
    for l, cat in enumerate(beta):
        if not np.isnan(cat):
            likelihood = t_dyn(l, cat, int(tree_topology[l]),hej_sibling(l, tree_topology))
    #likelihood = np.random.rand()
    # End: Example Code Segment

return likelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.3.")
    
    print("\n1. Load tree data from file and print it\n")
    
    #filename = "data/q2_3_small_tree.pkl"  # "data/q2_3_medium_tree.pkl", "data/q2_3_large_tree.pkl"
    #filename = "data/q2_3_medium_tree.pkl"
    filename = "data/q2_3_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()
    
    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files
    
    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
