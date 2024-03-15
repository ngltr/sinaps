#!/usr/bin/env python
# coding: utf-8

# # Data structure

# Follow a detailed description of the different features of the library.

# In[ ]:


import sinaps as sn


# ## Section
# 
# The class `Section` represents a section of neuron with uniform physical values
# 
# The characteristics of a section are :
# 
# * The length `L` in **μm**
# * The radius `a` in **μm**
# * The menbrane capacitance `C_m` in **μF/cm²**
# * The longitunal resistance `R_l` in **Ohm.cm**
# * The initial potential `V0` in **mV**

# ### Default parameters

# In[ ]:


sec0 = sn.Section(name="Sample section 1")
sec0


# ### Customized values

# Sinaps uses the [param](https://param.holoviz.org/) library. You can set custom parameters while creating the object:

# In[ ]:


sec1 = sn.Section(L=50,a=2,name="Sample section 2")
sec1


# You can also set the attribute values after the object is created:

# In[ ]:


sec1.R_l=100
sec1


# ## Channels
# 
# Ion channels can be added to a section.
# 
# There are two types of channels
# 
# * Point channels
# * Density channels
# 
# 
# 

# ### Density Channels
# Density channels are used to model channels that are distributed everywhere on a section. The current is given per unit of membrane surface.

# #### Leak Channel
# 

# In[ ]:


lc1=sn.channels.LeakChannel()
lc1


# In[ ]:


lc2=sn.channels.LeakChannel(
            Veq=10, #mV
            G_m= 1 #mS/cm²
            )
lc2


# In[ ]:


sec0.add_channel(lc1)
sec0


# ### Point Channels
# Point channels are used to model channel in specific location of the section, the given current is in pA (not relative to the section membrane surface).

# #### Constant current

# In[ ]:


pc=sn.channels.ConstantCurrent(1)


# In[ ]:


sec0.add_channel(pc,x = 0) #x relative position inside the section (0-1)
sec0


# The exhaustive list of implemented channels is described in the [API Reference](api_reference.html#channels)

# ## Neuron
# 
# The class `Neuron` represents a set of sections connected together

# In[ ]:


nrn=sn.Neuron()


# In[ ]:


nrn.add_section(sec0,0,1)
nrn.add_section(sec1,1,2)


# The structure of the neuron is stored in the attribute `sections` wich is a Dict with the section as keys and the nodes connected by the section as values (2-tuple) :

# In[ ]:


nrn.sections


# ### Accessing the sections
# By node index in the neuron structure. For example `nrn[i]` gives all the section connected to node i
# 

# In[ ]:


nrn[0]


# In[ ]:


nrn[1]


# By name

# In[ ]:


nrn['Sample section 2']


# Note that if sections have same names (if a part of their name is similar to the used keyname), a list of sections will be returned:

# In[ ]:


nrn['Sample section']


# You can change the parameters of multiples section at once :

# In[ ]:


nrn['Sample section'].C_m=1.5


# Access all the sections :

# In[ ]:


nrn[:]


# In[ ]:


nrn.plot()


# ## Simulation
# 
# The class simulation is linked to a specific neuron and is used to run voltage propagation simulation and electrodiffusion simulations with some custom spatial and time resolutions.
# The object is also storing the results of the simulation.
# 
# The solving is done using the [solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) function from the scipy library. Arguments from the solve_ivp function can be passed directly to sim.run.

# Create the simulation with neuron `nrn` and spatial resolution of `20`μm:

# In[ ]:


sim=sn.Simulation(nrn,dx=20)


# Run the simulation for timespan `0 - 300` ms: 

# In[ ]:


sim.run((0,300))


# ### Voltage

# Acces the results of the simulation, i.e. the potential at each time and position. They are stored in a [pandas](https://pandas.pydata.org/) Dataframe:

# In[ ]:


sim.V


# In[ ]:


sim['Sample section 2'].plot()


# In[ ]:


sim[:].plot()


# ### Currents

# The currents computed by the simulation environment are stored in a [pandas](https://pandas.pydata.org/) Dataframe. They are positive when they go toward inside.
# 
# The current can be accessed via their class:

# In[ ]:


sim.current(sn.channels.ConstantCurrent)


# In[ ]:


sim.current(sn.channels.ConstantCurrent)['Sample section 1'].hvplot()


# Getting all the channels in a simulation:

# In[ ]:


[ch.__name__ for ch in sim.channels]


# ## Species
# 
# The species are stored in the enum Class `Species`.
# This specific class is used to make the link in channels definition between currents and species flux.
# To simulate the concentration of a specific species, one first need to add it to the neuron, specifying the initial concentration and the diffusion coeficient. Ions that can be added through the API are: 
# 
# * Ca: calcium
# * K: potassium 
# * Na: sodium 
# * Buffer: any species, not charged
# * Anion: a negatively charged ion
# * Cation: a positively charger ion 

# In[ ]:


nrn.add_species(sn.Species.Ca,C0=2E-4, D=0.2)


# In[ ]:


sec0.C0


# In[ ]:


sec1.C0[sn.Species.Ca] = 1E-2


# In[ ]:


sim2=sn.Simulation(nrn,dx=20)


# Voltage simulation needs to be run first:

# In[ ]:


sim2.run((0,100))


# Then you can run the electro-diffusion simulation. The same way than for voltage simulations, the electro-diffusion simulations are run using the solve-ivp function from the scipy library. solve-ivp arguments can be passed directly to the run_diff function.

# In[ ]:


sim2.run_diff()


# As for the potential, the result of the simulation is stored in a [pandas](https://pandas.pydata.org/) Dataframe:

# In[ ]:


sim2.C


# In[ ]:


sim2.plot.C(sn.Species.Ca)


# If several channels of the same type are defined on a section, sim.plot.I will plot the sum of all the currents.

# In[ ]:


sim2.plot.C_field(sn.Species.Ca)

