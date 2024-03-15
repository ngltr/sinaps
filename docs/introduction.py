#!/usr/bin/env python
# coding: utf-8

# # Overview
# This is a short introduction to sinaps for new users.
# 
# The primary data structures of sinaps are:
# 
# * `Section` wich aims to represent a segment of a neuron with uniform physicals values
# * `Neuron` wich aims to represent a complete neuron is a directed graph whose edges are of `Section` type.
# * `Channels` wich aim to represent ionic channels with various dynamics. Several channels are already implemented, such as Hodgkin-Huxley channels, Pulse currents or AMPA receptors. <!-- Exhaustive list can be found in section XXX -->
# 
# The outputs of a simulation are:
# 
# * Electric potential, at each position in the neuron, for each time point
# * Electric currents computed from channels defined in the neuron, each postion, each time point
# * Concentration of species, at each position in the neuron, for each time point
# 
# 

# The package can be imported as follows:

# In[ ]:


import sinaps as sn


# ## Object creation
# Creating an empty `Neuron`:

# In[ ]:


nrn = sn.Neuron()


# Creating a `Section`, letting sinaps setting default attribute:

# In[ ]:


sec = sn.Section()


# Adding `HodgkinHuxley` [channels](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) (sodium, potassium and leak) to the newly created section:

# In[ ]:


sec.add_channel(sn.channels.Hodgkin_Huxley())


# Adding a Hodgkin-Huxley type calcium channel:

# In[ ]:


sec.add_channel(sn.channels.Hodgkin_Huxley_Ca())


# 
# Adding a `PulseCurrent` channel with a current of `200` pA, between t =`2` and t=`5` ms, at the beginning of the section (position `0`):

# In[ ]:


sec.add_channel(sn.channels.PulseCurrent(200,2,5),0)


# Adding the section `s` to the neuron `nrn` as an edge between nodes `0` and `1` :

# In[ ]:


nrn.add_section(sec,0,1)


# Adding two new sections
# * `sec2` with a radius of `2` μm
# * `sec3` with a length of `200` μm

# In[ ]:


sec2 = sn.Section(a = 2)
sec3 = sn.Section(L = 200)
nrn.add_sections_from_dict({
    sec2:(1,2),
    sec3:(1,3)
})
    


# Plotting the neuron structure:

# In[ ]:


nrn.plot()


# Adding calcium ions in the model:

# In[ ]:


nrn.add_species(sn.Species.Ca,C0=2E-4, D=100) # For the sake of the example, we increase the calcium diffusion coefficient to speed-up its dynamics, in order to observe variations within 50 ms. 


# ## Running simulation

# Creating a `Simulation` of neuron `nrn` with spatial resolution `10` μm:

# In[ ]:


sim = sn.Simulation(nrn,dx=10)


# Running the simulation for timespan `0` to `50` ms. 

# In[ ]:


sim.run((0,50))


# Results of the simulation are stored as [pandas](https://pandas.pydata.org/) Dataframe:

# In[ ]:


sim.V


# ## Viewing results

# Plotting up to 10 curves distributed evenly on the neuron:

# In[ ]:


sim.plot()


# Plotting the Hodgkin-Huxley currents:

# In[ ]:


sim.plot.I(sn.channels.Hodgkin_Huxley)


# Getting a field view of the potential:

# In[ ]:


sim.plot.V_field()


# Running the electrodiffusion part:

# In[ ]:


sim.run_diff(max_step=1) 


# Plotting the calcium concentration dynamics

# In[ ]:


sim.plot.C(sn.Species.Ca)


# In[ ]:


sim.plot.C_field(sn.Species.Ca)

