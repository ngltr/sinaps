#!/usr/bin/env python
# coding: utf-8

# # Customizing channels

# In[ ]:


import sinaps as sn


# In[ ]:


import numpy as np


# To implement a channel C, it is necessary to implement:
# 
#  * a static method  C._I(V,t,**st_vars,**params) returning the net  current towards inside
#   ([pA] for point channel and [pA/μm2] for density channels) of the mechanism with
#  V the voltage (:array) and **st_vars the state variable of the mechanism (:array)
#  * If there are state variables (ions), a static method C._dS(V,t,**st_vars,**params) returning a
#   tuple with the differential of each state variable and a function S0 returning a tuple with the initial value for each state variable
#  * a static method C._J(ion,V,t,**st_vars,**params) returning the flux of each species due to the current, in mM/ms for point channels, and in mM/ms/μm^{2} ofr density channels.
# 

# In this example we will create one additionals channel, a NMDA receptor, with 15% of the current carried by calcium ions.
# We need to:
# 
# * subclass the Channel Class
# * define the _init function
# * define the static method C._I, that computes the current, in pA for point channels, and in pA/μm^2 for density channels
# * define the static method C._J., that computes the corresponding the flux of ions, in mM/ms. The conversion from pA to mM/ms requires to divide by 96.48533132838746, and by the valence of the ion.

# In[ ]:


class NMDAR(sn.Channel):
    """Point channel with a NMDAr-type current starting at time t0, 
voltage-dependent flow of sodium (Na+) and small amounts of calcium (Ca2+) ions into the cell and potassium (K+) out of the cell.
    """
    param_names = ('t0','gnmda','tnmda1','tnmda2','V_nmda')

    def __init__(self, t0, gnmda=0.02,tnmda1=11.5,tnmda2=0.67,V_nmda=75):
        """Point channel with a NMDAr-type current starting at time t0 [pA]
            t0: start of the current [ms]
            gnmda: max conductance of NMDAr [nS]
            tnmda1: NMDAr time constant [ms]
            tnmda2: NMDAr time constant [ms]
            V_nmda: NMDAr Nernst potential [mV]
        """
        self.params={'t0' : t0,
                     'gnmda' : gnmda,
                     'tnmda1' : tnmda1,
                     'tnmda2' : tnmda2,
                     'V_nmda': V_nmda,
                    }

    @staticmethod
    def _I(V,t,
           t0,gnmda,tnmda1,tnmda2,V_nmda):
        return -((t <= t0+50) & (t >= t0))*gnmda*(np.exp(-np.abs(t-t0)/tnmda1)-np.exp(-np.abs(t-t0)/tnmda2))/(1+0.33*2*np.exp(-0.06*(V-65)))*(V-V_nmda)

    
    @staticmethod
    def _J(ion,V,t,
           t0,gnmda,tnmda1,tnmda2,V_nmda):
        """
        Return the flux of ion [aM/ms/um2] of the mechanism towards inside.
        """
        if ion is sn.Species.Ca:
            return ((t <= t0+50) & (t >= t0)) *np.maximum(-gnmda*(np.exp(-np.abs(t-t0)/tnmda1)-np.exp(-np.abs(t-t0)/tnmda2))/(1+0.33*2*np.exp(-0.06*(V-65)))*(V-V_nmda)/96.48533132838746/2*0.15,0)
        else:
            return 0 *V


# ### Setting up the channels

# In[ ]:


nrn = sn.Neuron([(0,1),(1,2),(1,3),(3,4)])


# In[ ]:


nrn[0]


# In[ ]:


nrn[:].clear_channels()
nrn[:].add_channel(sn.channels.LeakChannel())
nrn[:].add_channel(sn.channels.Hodgkin_Huxley_Ca())

nrn[0].add_channel(NMDAR(0.5,gnmda=20),0) #


# In[ ]:


# Shortcuts
Ca = sn.Species.Ca


# In[ ]:


nrn.add_species(Ca, D=50, C0=0)


# ### Running the simu

# In[ ]:


# Initialisation of the simulation
sim=sn.Simulation(nrn,dx=100)


# In[ ]:


# Runing the simulation
sim.run((0,100))


# ### Plots

# In[ ]:


# Plotting the potential
sim.plot()


# In[ ]:


sim.plot.I(NMDAR)


# ### Running the calcium dynamics

# In[ ]:


# Running the chemical reactions part   
sim.run_diff(max_step=1)


# ### Plots

# In[ ]:


sim[:].plot.C(Ca)

