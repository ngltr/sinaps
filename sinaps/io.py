import re

from sinaps import Neuron, Section

def read_comp_imaging(filename):
    """Return a Neuron from a Data Comprehensive file"""
    # Read such as nodeId contains list of all tree edges
    #[Edge1_id_node parent,Edge1_id_node_child,Edge2_id_node parent,Edge2_id_node_child,...]
    with open(filename) as f:
        nodeIDs=re.findall(r"NodeID:(-?\d+)\r?\n[PN]",f.read())
    # 2.convert string to int
    nodeIDs=list(map(int,nodeIDs))
    #reaarange values to get list of edges
    edges=[(nodeIDs[k+1],nodeIDs[k]) for k in range(0,len(nodeIDs)-1,2)]

    N=Neuron()
    for parent,child in edges:
        N.add_section(Section(),parent,child)
    N.__traversal_source__ = -1

    return N
