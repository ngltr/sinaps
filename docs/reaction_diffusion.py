#!/usr/bin/env python
# coding: utf-8

# # Reaction Diffusion

# In this example we consider a model where calcium ions interact with a buffer. We also add calcium extrusion.

# In[ ]:


import sinaps as sn


# ## Initialization

# Creating a sample neuron:

# In[ ]:


nrn = sn.Neuron([(0,1),(1,2),(1,3),(3,4)])
nrn.plot()


# Settings channels: note that the `Hodgkin_Huxley_Ca` channel needs to be added to account for calcium entry through voltage-gated calcium channels. The `Hodgkin_Huxley` channels only contains sodium, potassium and leak channels.

# In[ ]:


nrn[0].clear_channels()#Reset all channels, useful if you re-run this cell (does nothing the first time)
nrn[0].add_channel(sn.channels.PulseCurrent(500,2,3),0)
nrn[0].add_channel(sn.channels.Hodgkin_Huxley())
nrn[0].add_channel(sn.channels.Hodgkin_Huxley_Ca(gCa=14.5,V_Ca=115)) 


# In[ ]:


sim = sn.Simulation(nrn,dx=10)


# In[ ]:


sim.run((0,100))


# In[ ]:


sim.plot()


# ## Chemical reactions

# Ion species can be added to the model via the function `add_species`

# nrn.add_species(sn.Species.Ca)

# Chemical reactions are defined with the reaction equation and the reaction
# rates:
# $$ Ca + Bf \rightleftharpoons_{k_2}^{k_1} BfCa $$
# 
# 
# Reactions are simulated during the electrodiffusion simulation.
# 
# Note that Species are automatically added to the neuron if they are not already
# present.

# In[ ]:


#Reset reaction / not useful the first time
nrn.reactions=[]


# In[ ]:


#shortcuts
Ca=sn.Species.Ca
BF=sn.Species.Buffer
BFB=sn.Species.Buffer_Ca


# In[ ]:


nrn.add_reaction(
    {Ca:1,
     BF:1},
    {BFB:1},
    k1=0.2,
    k2=0.1)


# The calcium extrusion is modeled using the equation:
# $$ Ca  \rightleftharpoons^{k_1} \emptyset $$

# In[ ]:


#Calcium extrusion
nrn.add_reaction(
    {Ca:1},
    {},
    k1=0.05,k2=0)


# Initial concentrations can be set:

# In[ ]:


#Calcium initial concentration
for s in nrn.sections:
    s.C0[Ca]=10**(-4)
    s.C0[BF]=2*10**(-3)


# ## Running electro diffusion simulatiom

# In[ ]:


sim.run_diff(max_step=1)


# ## Plotting results

# In[ ]:


sim.plot.C(Ca)


# In[ ]:


sim.plot.C(BFB)


# In[ ]:


(sim.plot.V().opts(width=900,height=200) 
 + sim.plot.C(Ca) .opts(width=900,height=200) 
 + sim.plot.C(BFB).opts(width=900,height=200) 
).cols(1).opts(plot={'shared_axes':False})


# In[ ]:


sim.plot.C_field(Ca)


# In[ ]:




