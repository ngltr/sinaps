#!/usr/bin/env python
# coding: utf-8

# # Plotting

# `Sinaps` includes a convenient way to directly plot a neuron geometry and simulation results, through the .plot accesor. This is based on the [holoview](http://holoviews.org/) library.

# To illustrate the plotting tool, we first create a simulation of voltage propagation and calcium electrodiffusion, in a simple neuron. Calcium dynamics includes buffering and extrusion.
# 

# In[ ]:


import sinaps as sn

# Defining the nueron
nrn = sn.Neuron([(0,1),(1,2),(1,3),(3,4)])

# Setting up channels
nrn[0].clear_channels()#Reset all channels, useful if you re-run this cell (does nothing the first time)
nrn[0].add_channel(sn.channels.PulseCurrent(200,2,3),0)
nrn[0].add_channel(sn.channels.PulseCurrent(200,22,23),0)
nrn[0].add_channel(sn.channels.Hodgkin_Huxley())
nrn[0].add_channel(sn.channels.Hodgkin_Huxley_Ca(gCa=14.5,V_Ca=115)) 

# Defining and running the simulation
sim = sn.Simulation(nrn,dx=10)
sim.run((0,60))

#Setting initial concentrations
for s in nrn.sections:
    s.C0[sn.Species.Ca]=10**(-4)
    s.C0[sn.Species.Buffer]=2*10**(-3)
    s.C0[sn.Species.Buffer_Ca]=0

# Adding chemical reactions: claicum binding to a buffer, and calcium extrusion
nrn.add_reaction(
    {sn.Species.Ca:1,
     sn.Species.Buffer:1},
    {sn.Species.Buffer_Ca:1},
    k1=0.2,
    k2=0.1)

nrn.add_reaction(
    {sn.Species.Ca:1},
    {},
    k1=0.05,k2=0)

# Running electro-diffusion simulations
sim.run_diff(max_step=1)


# ## Neuronal geometry

# In[ ]:


import hvplot
hvplot.__version__


# To plot the neuron geometry:

# In[ ]:


nrn.plot()


# ## Potential

# `sim.plot.V` plots the voltage dynamics at nodes evenly distributed on the neuron. Default number of curves is 10, change number with `max_plot`.

# In[ ]:


sim.plot.V(max_plot=2)


# Specific sections can be accesed through their names:

# In[ ]:


sim['Section0000'].plot()


# Holoview or matplotlib can be used directly through the [pandas](https://pandas.pydata.org/) Dataframe structure of the data: 

# In[ ]:


sim.V


# In[ ]:


sim.V.loc[:,'Section0000'][25].hvplot()


# is equivalent to:

# In[ ]:


sim['Section0000'].V.loc[:,25].hvplot()


# The name of the sections can be accessed through the columns function. See the [pandas](https://pandas.pydata.org/) doc for more features.

# In[ ]:


sim.V.columns


# In[ ]:


sim.V[('Section0000', 25.0)].plot()


# Field plots are also acccessible through the API. The time can be zoomed through the command `time`. See the [holoview](https://holoviews.org/) doc for more features. 

# In[ ]:


sim.plot.V_field(time=(0,30))


# To help navigating in the field plot, the neuronal geometry is plotted on the left, with a colorbar legend mapping the linear position on the field to the geometry. In a live notebook, the view is interactive. Passing the mouse on the field plot will enhance the corresponding section on the neuronal geometry. 
# 
# Setting the parameter `neuron` to False displays only the field plot.

# ## Plotting the currents

# The currents computed by the simulation environment are stored in a [pandas](https://pandas.pydata.org/) Dataframe. They are positive when they go toward inside.
# 
# `sim.plot.I(ChannelClass)` plots the currents for the channels of type ChannelClass

# In[ ]:


sim.plot.I(sn.channels.Hodgkin_Huxley)


# The field plots are also accessible:

# In[ ]:


sim.plot.I_field(sn.channels.Hodgkin_Huxley)


# If several channels of the same type are defined on a section, sim.plot.I will plot the sum of all the currents:

# In[ ]:


sim['Section0000'].plot.I(sn.channels.PulseCurrent)


# Holoview or matplotlib can be used directly through the [pandas](https://pandas.pydata.org/) Dataframe structure of the data: 

# In[ ]:


sim.current(sn.channels.Hodgkin_Huxley) 


# In[ ]:


sim.current(sn.channels.PulseCurrent)['Section0000'].hvplot()


# ## Plotting the species dynamics

# The species concentration dynamics computed by the simulation environment are stored in a [pandas](https://pandas.pydata.org/) Dataframe.
# 
# `sim.plot.C(Species)` plots the concentration of Class Species:

# In[ ]:


sim.plot.C(sn.Species.Ca)


# In[ ]:


sim['Section0000'].plot.C(sn.Species.Ca)


# The species dynamics are stored in a [pandas](https://pandas.pydata.org/) Dataframe, and are accessible via their class:

# In[ ]:


sim.C[sn.Species.Ca]


# In[ ]:


sim.C[sn.Species.Ca].loc[:,'Section0000'][25].hvplot()


# is equivalent to:

# In[ ]:


sim['Section0000'].C(sn.Species.Ca).loc[:,25].hvplot()


# In[ ]:


sim['Section0000'].C(sn.Species.Buffer_Ca).loc[:,5].hvplot()

