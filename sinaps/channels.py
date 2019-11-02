# coding: utf-8
import numpy as np
from quantiphy import Quantity

from .core.model import Channel
from .ions import ion_Ca

ion_Ca



"""
Units :

Voltage : mV
Time : ms
Distance : μm
Resistance : GΩ
Conductance : nS
Capacitance : pF
Current : pA

To implement a channel C, it is necessary to implement  :
 - a function C.I(V,**st_vars) returning the net  current towards inside
  ([A] for point channel and [pA/μm2] for density channels ) of the mechanism with
 V the voltage (:array) and **st_vars the state variable of the mechanism (:array)
  - If there is state variables a function C.dS(V,**st_vars) returning a
  tuple with the differential of each state variable
   and a function S0 returning a tuple with the initial value for each state
   variable
"""


class ConstantCurrent(Channel):
    """Point channel with a constant current
       current [pA]
    """
    def __init__(self,current):
        super().__init__()
        self.current = current


    def I(self,V,*args):
        return self.current


    def __repr__(self):
        return "ConstantCurrent(I={})".format(
            Quantity (self.current*1E-12,'A'))

class HeavysideCurrent(Channel):
    """Point channel with a constant current between a time period

    """
    def __init__(self,current,t0,tf):
        """Point channel with a constant current
            current [pA]
            t0 : start of the current [ms]
            tf : end of the current [ms]
        """
        super().__init__()
        self.current = current
        self.t0= t0
        self.tf = tf


    def I(self,V,S,t):
        if t <= self.tf and t >= self.t0:
            I = self.current
        else :
            I = 0
        return I

class LeakChannel(Channel):
    """Leak channel
        I = (V-Veq) / Rm
    """
    def __init__(self,Veq=0,G_m=0.3):
        """
            Veq : [mV] Equilibrium potential for the leak channel
            G_m : [mS/cm2] Resistance of the menbrane (leak channel)
        """
        super().__init__()
        self.Veq = Veq
        self.R_m = 1/G_m * 100 # conversion to GΩ.μm2: 1/(1 mS/cm2) = 100 GΩ.μm2


    def I(self,V,*args):
        """
        Return the net surfacic current [pA/um2] of the mechanism towards inside
        """
        return (self.Veq - V) / self.R_m

    def __repr__(self):
        return "LeakChannel(Veq={}, G_m={})".format(
            Quantity (self.Veq*1E-3,'V'),
            Quantity (1/self.R_m*1E-1,'S/cm²'))


class Hodgkin_Huxley(Channel):
    """Channel Hodgkin Huxley type

    """

    def __init__(self, gNa=120, V_Na=115, gK =36, V_K=-12, gL=0.3 ,V_L=10.6):
        """Channel Hodgkin Huxley type
            gNa : conductance of sodium channel [mS/cm2]
            V_Na : Equilibrium potential of sodium channel [mV]
            gK :  conductance of potasium channel [mS/cm2]
            V_K : Equilibrium potential of potasium channel [mV]
            gK :  conductance of leak channel [mS/cm2]
            V_L : Equilibrium potential of leak channel [mV]
        """
        self.nb_var = 1
        self.gNa = gNa / 100 # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
        self.V_Na = V_Na
        self.gK = gK / 100 # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
        self.V_K = V_K
        self.gL = gL / 100 # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
        self.V_L = V_L


    def I(self,V,n,t):
        """
        Return the net surfacic current [pA/um2] of the mechanism towards inside
        """
        alpha_m = (2.5-0.1*V)/(np.exp(2.5-0.1*V)-1)
        beta_m = 4*np.exp(-V/18)
        m = alpha_m/(alpha_m + beta_m);
        h = 0.89 - 1.1 * n
        I_Na = self.gNa * m**3 * h * (V -  self.V_Na)
        I_K = self.gK * n**4 * (V - self.V_K)
        I_L = self.gL * (V - self.V_L)
        return - I_Na - I_K - I_L


    def dS(self, V, n):
        dn = 0.1 * (1 - 0.1 * V) * (1-n)/(np.exp(1-0.1*V)-1) - 0.125 * np.exp(-V/80)*n
        return dn

    def S0(self):
        """Return the initial value for the state variable
        n = 0.316
        """
        return 0.3162

class Hodgkin_Huxley_Ca(Channel):
    """ Ca channel to add to Hodgkin Huxley
    """
    def __init__(self, gCa=14.5E-9, V_Ca=140):
        """Channel Hodgkin Huxley type
            gCa : conductance of calcium channel [mS/cm2]
            V_Ca : Equilibrium potential of calcium channel [mV]
        """
        self.nb_var = 2
        self.gCa = gCa / 100 # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
        self.V_Ca = V_Ca


    def I(self,V,m,h,t):
        """
        Return the net surfacic current [pA/um2] of the mechanism towards inside
        """
        I_Ca = self.gCa * m**3 * h * (V - self.V_Ca)
        return -I_Ca


    def dS(self, V, m, h):
        dm = 1/1.3 * (1/(1 + np.exp(-V+102)) - m)
        dh = 1/10 * (1/(1 + np.exp(-V+24)) - h)
        return dm, dh

    def S0(self):
        """Return the initial value for the state variable
        m = 0
        h = 1
        """
        return 0, 1
