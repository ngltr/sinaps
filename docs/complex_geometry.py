#!/usr/bin/env python
# coding: utf-8

# # Complex Geometry
# 
# Example notebook to load a complex geometry from a swc file, and add synapses on it.
# 

# In[ ]:


import sinaps as sn

import numpy as np
import holoviews as hv
import random


# ## Creating Neuron structure from swc file

# We choose a neuron from the [neuromorpho](http://neuromorpho.org) database:

# In[ ]:


#filename = "http://neuromorpho.org/dableFiles/chang/CNG%20version/V247fs-MT-Untreated-56.CNG.swc" 
filename = "https://raw.githubusercontent.com/ngltr/sinaps/master/docs/sample_data/V247fs-MT-Untreated-56.CNG.swc"


# Use the function read_swc from io module to create a neuron from this file:

# In[ ]:


nrn = sn.io.read_swc(filename)
nrn.plot()


# ## Accessing data

# The neuron structure is stored using the [networkx](https://networkx.org) library, and you can access this underlying structure using:

# In[ ]:


nrn.graph


# For example :

# In[ ]:


nrn.graph.edges


# In[ ]:


nrn.graph.nodes


# You can access the section corresponding to an edge in the graph structure using the `section` attribute of edges

# In[ ]:


nrn.graph[1][2]['section']


# ### Setting custom radius

# The swc file contains the radius for each node of the dendrite. In the following, we redefine the radius of each section, assuming that this radius decreases at each bifurcation.
# We define for that a recursive function. 

# In[ ]:


def set_radius(G,dep,a,factor):
    if G.degree(dep)>1:
        a=max(a*(G.degree(dep)-1)**(factor),0.1)
        for node in G[dep]:
            s = G[dep][node]['section']
            if s.a == 1000:# 1000 = marker
                s.a=a
                set_radius(G,node,a,factor)


# In[ ]:


nrn['dendrite'].a = 1000 #initialisation


# In[ ]:


# Setting radius of each node
set_radius(nrn.graph,1,5,-0.5) 


# In[ ]:


nrn.plot()


# The neuron is plotted using the coordinates of the swc file, but we can force the recalculation of the layout to optimize the graph visualization:

# In[ ]:


nrn.plot.layout(force=True)
nrn.plot().opts(node_size=0)


# By defaut, the layout is calculated using the Kamada-Kawai algorithm but you can use any other node positioning algorithm setting the `layout`parameter

# ## Modeling synapses
# 
# ### Selecting leaves
# 
# We now randomly choose 10 leaves (excluding the soma) where we put synpases: we first remove the Hodgkin-Huxley channels, and add AMPA receptors.

# We first find the node indices of the soma, to exclude it:

# In[ ]:


s = nrn['soma']
[nrn.sections[s] for s in s]


# Then we use the `nrn.leaves` function to find leaves of the neuron structure

# In[ ]:


leaves = nrn.leaves() # Finding the leaves
leaves.remove(2) # Removing the soma
leaves.remove(3) # Removing the soma
stim_leaves = random.sample(leaves,10) # getting randomly 10 leaves 
stim_sec = nrn[stim_leaves] # list of the stimulated sections


# In[ ]:


stim_sec.a


# `stim_sec` is a list of sections, of type section_list, on which one can operate. For example, `stim_sec.a` will give the radius of all sections in `stim_sec`. It also allows the operations on all the series in list, such as adding channels, or changing the radius.

# ### Setting up the channels
# We add Hodgkin Huxley channel everywhere in the neuron
# and AMPAR channel on previously chosen synapses.
# 
# The `nrn[:]` command select all sections and allws to see or set parameters in one command
# 

# In[ ]:


nrn[:].clear_channels() #reset all channels
nrn[:].add_channel(sn.channels.Hodgkin_Huxley()) # add HH channel everywhere
stim_sec.clear_channels() #remove all channels on stim_sec sections 
stim_sec.add_channel(sn.channels.AMPAR(0.5,gampa=0.5),1) # add NMDAR channel on stim_sec sections


# ### Plotting stimulated leaves
# 
#   

# In[ ]:


nrn.add_node_data(type=0) #default node
nrn.add_node_data(stim_leaves,type=2) # synapses are put in a different color, different size.
plt = nrn.plot()
plt.opts(
    node_size = hv.dim('type')*4,
    node_color = 'red', 

)


# ## Running the simulation

# In[ ]:


# Initialisation of the simulation
sim=sn.Simulation(nrn,dx=50)


# In[ ]:


# Runing the simulation
sim.run((0,100))


# ## Plotting

# In[ ]:


# Plotting the potential
sim.plot()


# In[ ]:


sim.plot.I(sn.channels.Hodgkin_Huxley)


# In[ ]:


sim.plot.V_field()


# In[ ]:


# Extracting some sections
sim['soma'].plot()


# In[ ]:


sim.plot.I_field(sn.channels.AMPAR)


# In[ ]:




