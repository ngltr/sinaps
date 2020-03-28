#!/usr/bin/env python
import numpy as np
import re
from neuron import h, gui


#################### Converting Kelly's file  #############################
filename=('interp-neuron-.txt')


with open(filename) as f:
    nodeIDs=re.findall(r"NodeID:(-?\d+)\r\n[PN]",f.read())
    #nodeId contains list of all tree edges [Edge1_id_node parent,Edge1_id_node_child,Edge2_id_node parent,Edge2_id_node_child,...]

    nodeIDs=map(int,nodeIDs) #convert string to int
    edges=[(nodeIDs[k+1],nodeIDs[k]) for k in range(0,len(nodeIDs)-1,2)] #reaarange values to get list of edges


#################### Defining the sections  #############################
n=max(max(edges))+1 #number of nodes
sections=[]
soma=h.Section(name="soma")
for i in xrange(n):
    sections.append(h.Section(name="node{}".format(i)))

for parent,child in edges:
    if parent > -1:
        sections[child].connect(sections[parent](1))
    else:
        sections[child].connect(soma(1))#connect to the soma
