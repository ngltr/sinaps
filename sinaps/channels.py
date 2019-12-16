# coding: utf-8
import numpy as np
from quantiphy import Quantity

from .core.model import Channel

from .species import Species

from numba import jit

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
 - a function C._I(V,t,**st_vars,**params) returning the net  current towards inside
  ([A] for point channel and [pA/μm2] for density channels ) of the mechanism with
 V the voltage (:array) and **st_vars the state variable of the mechanism (:array)
 - If there is state variables a function C._dS(V,t,**st_vars,**params) returning a
  tuple with the differential of each state variable
   and a function S0 returning a tuple with the initial value for each state
   variable
"""


class ConstantCurrent(Channel):
    """channel with a constant current
       current [pA] (point channel)
       or [pA/um2] (density channel)
    """
    param_names=('current',)

    def __init__(self,current):
        self.params = {'current' : current}

    @staticmethod
    def _I(V,t,current):
        return current


    def __repr__(self):
        return "ConstantCurrent(I={})".format(
            Quantity (self.params['current']*1E-12,'A{}'.format(self.density_str())))



class HeavysideCurrent(Channel):
    """Point channel with a constant current between a time period

    """
    param_names = ('current','t0','tf')

    def __init__(self,current,t0,tf):
        """Point channel with a constant current
            current [pA]
            t0 : start of the current [ms]
            tf : end of the current [ms]
        """
        self.params={'current': current,
                    't0' : t0,
                    'tf' : tf,
                    }

    @staticmethod
    def _I(V,t,
           current,t0,tf):
        return ((t <= tf) & (t >= t0)) * current

class LeakChannel(Channel):
    """Leak channel
        I = (V-Veq) / Rm
    """
    param_names=('Veq','R_m')

    def __init__(self,G_m=0.3,Veq=0):
        """
            Veq : [mV] Equilibrium potential for the leak channel
            G_m : [mS/cm2] Resistance of the menbrane (leak channel)
        """
        self.params = {'Veq' : Veq,
                       'R_m' : 1/G_m * 100}
                       #conversion to GΩ.μm2: 1/(1 mS/cm2) = 100 GΩ.μm2

    @staticmethod
    def _I(V,t,
           Veq,R_m):
        """
        Return the net surfacic current [pA/um2] of the mechanism towards inside
        """
        return (Veq - V) / R_m

    def __repr__(self):
        return "LeakChannel(Veq={}, G_m={})".format(
            Quantity (self.params['Veq']*1E-3,'V'),
            Quantity (1/self.params['R_m']*1E-1,'S/cm²'))


class Hodgkin_Huxley(Channel):
    """Channel Hodgkin Huxley type

    """
    nb_var = 1
    param_names=('gNa','V_Na','gK','V_K','gL','V_L')

    def __init__(self, gNa=120, V_Na=115, gK =36, V_K=-12, gL=0.3 ,V_L=10.6):
        """Channel Hodgkin Huxley type
            gNa : conductance of sodium channel [mS/cm2]
            V_Na : Equilibrium potential of sodium channel [mV]
            gK :  conductance of potasium channel [mS/cm2]
            V_K : Equilibrium potential of potasium channel [mV]
            gK :  conductance of leak channel [mS/cm2]
            V_L : Equilibrium potential of leak channel [mV]
        """
        self.params={'gNa' : gNa / 100, # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
                    'V_Na' : V_Na,
                    'gK' : gK / 100, # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
                    'V_K' : V_K,
                    'gL' : gL / 100, # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
                    'V_L' : V_L,
                    }


    @staticmethod
    def _I(V,n,t,
           gNa,V_Na,gK,V_K,gL,V_L):
        alpha_m = (2.5-0.1*V)/(np.exp(2.5-0.1*V)-1)
        beta_m = 4*np.exp(-V/18)
        m = alpha_m/(alpha_m + beta_m);
        h = 0.89 - 1.1 * n
        I_Na = gNa * m**3 * h * (V -  V_Na)
        I_K = gK * n**4 * (V - V_K)
        I_L = gL * (V - V_L)
        return - I_Na - I_K - I_L

    @staticmethod
    def _dS(V, n, t,
            gNa,V_Na,gK,V_K,gL,V_L):
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
    nb_var = 2
    param_names=('gCa','V_Ca')

    def __init__(self, gCa=14.5E-9, V_Ca=140):
        """Channel Hodgkin Huxley type
            gCa : conductance of calcium channel [mS/cm2]
            V_Ca : Equilibrium potential of calcium channel [mV]
        """
        self.params={'gCa': gCa / 100, # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
                    'V_Ca' : V_Ca}

    @staticmethod
    def _I(V,m,h,t,
           gCa,V_Ca):
        """
        Return the net surfacic current [pA/um2] of the mechanism towards inside
        """
        I_Ca = gCa * m**3 * h * (V - V_Ca)
        return -I_Ca

    @staticmethod
    def _J(ion,V,m,h,t,
           gCa,V_Ca):
        """
        Return the flux of ion [aM/ms/um2] of the mechanism towards inside
        """
        if ion is Species.Ca:
            return gCa * m**3 * h * (V - V_Ca) /96.48533132838746/2
        else:
            return 0 * V

    @staticmethod
    def _dS(V, m, h, t,
            gCa, V_Ca):
        dm = 1/1.3 * (1/(1 + np.exp(-V+102)) - m)
        dh = 1/10 * (1/(1 + np.exp(-V+24)) - h)
        return dm, dh

    def S0(self):
        """Return the initial value for the state variable
        m = 0
        h = 1
        """
        return 0, 1


def custom(func,name="Custom channel"):
    C=type(name,(Channel,),{"_I":func})
    return C()
