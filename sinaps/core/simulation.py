# coding: utf-8
from functools import lru_cache
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, dia_matrix
from scipy import interpolate
from numba import jit

from tqdm import tqdm


class Simulation:
    """This class represent a simulation
    usage :
    S=Simulation(N,dx) with N type:Neuron and dx spatial_resolution [um]
    S.run(tnrange)  with tnrange [ms]

    To record the concentration of one ion, one must run :
    S.record_ion(ion) with ion of type Ion


    """


    def __init__(self, neuron, dx=1):
        self.N = neuron
        self.idV,idS0 = self.N.init_sim(dx)
        self.idS = idS0 + self.idV[-1] + 1
        self.Cm1 = 1/self.N.capacitance_array()
        self.Vol1 = 1/self.N.volume_array()
        self.G = csr_matrix(self.N.conductance_mat())#sparse matrix format for
                                                     #efficiency
        self.k_c = csr_matrix(self.N.connection_mat())

        self.V_S0 = np.concatenate([self.N.V0_array(), self.N.S0_array()])

        self.ions = 0
        self.C = dict()
        self.sol_diff = dict()

    def record_ion(self,ion):
        self.ions.append(ion)
        self.V_S0 = np.concatenate([self.V_S0, np.zeros(self.N.nb_comp)])

    def run(self,t_span,method='BDF',atol = 1.49012e-8,**kwargs):
        """Run the simulation
        t : array
        A sequence of time points (ms) for which to solve the system
        """
        tq=tqdm(total=t_span[1]-t_span[0],unit='ms')
        sol=solve_ivp(lambda t, y:Simulation.ode_function(y,t,self.idV,self.idS,
                                                self.Cm1,self.G,self.k_c,self.N,
                                                tq,t_span),
                      t_span,
                      self.V_S0,
                      method=method,
                      atol = atol,
                       **kwargs)
        tq.close()

        sec,pos = self.N.indexV()
        df = pd.DataFrame(sol.y[:self.N.nb_comp,:].T ,
                          sol.t ,
                          [sec, pos])
        df.columns.names = ['Section','Position (μm)']
        df.index.name='Time'
        self.V = df + self.N.V_ref
        self.sol = sol
        #sec,pos = self.N.indexS()
        df = pd.DataFrame(sol.y[self.idS,:].T ,
                          sol.t )#,[sec, pos * 1E6])
        #df.columns.names = ['Section','Channel','Variable,''Position (μm)']
        df.index.name='Time (ms)'
        self.S=df
        self.t_span=t_span

    @staticmethod
    def ode_function(y,t,idV,idS,Cm1,G,k_c,neuron,tq=None,t_span=None):
        """this function express the ode problem :
        dy/dt = f(y)

        y is a vector containing all state variables of the problem

        V=y[idV] : voltage of each node V=0 at resting potential
                    size n + m where n is the total number of compartiment and
                    m the total number of connecting nodes
        S=y[idS] : other states variables related to the ion channels

        Cm1 : Inverse of the capacitance of the membrane for each compartiment size n
        G : Conductance matrix size n * n+m
        k_c : Connection matrix size m * n


        Voltage equation  for compartiment is given by :
        dV/dt = 1/Cm (G.V + Im )

        Voltage equation for connecting nodes linearly dependent of
        compartment nodes:
        dV/dt = k_c.dV

        neuron : type Neuron contains the information about the behaviour
         of the ion channels in order to compute I
        """

        V = y[idV]
        S = y[idS]
        I = neuron.I(V, S, t) #current of active ion channels from outisde to inside
        dVi = Cm1 * (G @ V + I) #dV/dt for  compartiment
        dVo = k_c @ dVi #dV/dt for connecting nodes
        dS = neuron.dS(V,S)

        #Progressbar
        if not (tq is None):
            n=round(t-t_span[0],3)
            if n>tq.n:
                tq.update(n-tq.n)

        return np.concatenate([dVi,dVo,dS])


    def run_diff(self,ion,temperature=310,method='RK45',atol = 1.49012e-8,**kwargs):
        """Run the simulation diffusion for ion ion
        The simulation for the voltage must have been run before
        """
        tq=tqdm(total=self.t_span[1]-self.t_span[0],unit='ms')
        V=interpolate.interp1d(self.sol.t,self.sol.y[self.idV],
                                fill_value='extrapolate')
        S=interpolate.interp1d(self.sol.t,self.sol.y[self.idS],
                                fill_value='extrapolate')
        Simulation._flux.cache_clear()
        Simulation._difus_mat.cache_clear()

        sol=solve_ivp(lambda t, y:Simulation.ode_diff_function(y,t,V,S,
                                                self.Vol1,self.k_c,self.N,ion,
                                                temperature,
                                                tq,self.t_span),
                      self.t_span,
                      self.N.C0_array(ion),
                      method=method,
                      atol = atol,
                       **kwargs)
        tq.close()

        sec,pos = self.N.indexV()
        df = pd.DataFrame(sol.y[:self.N.nb_comp,:].T ,
                          sol.t ,
                          [sec, pos])
        df.columns.names = ['Section','Position (μm)']
        df.index.name='Time'
        self.C[ion] = df
        self.sol_diff[ion] = sol

    @staticmethod
    def ode_diff_function(C,t,V,S,Vol1,k_c,neuron,ion,T,tq=None,t_span=None):
        """this function express the ode problem :
        dy/dt = f(y)

        y is a vector containing all state variables of the problem

        C=y : concentration of each node
                    size n + m where n is the total number of compartiment and
                    m the total number of connecting nodes

        V : Voltage array (previously computed)
        S : Status variables of the channels (previously computed)

        k_c : Connection matrix size m * n

        T: Temperature (Kelvin)


        Concentration equation  for compartiment is given by :
        dC/dt = 1/Vol(D V + Jc )
        with D : Electrodiffusion matrix size n * n+m
        Vol : Volume μm^3

        Concentration equation for connecting nodes linearly dependent of
        compartment nodes:
        dC/dt = k_c.dC

        neuron : type Neuron contains the information about the behaviour
         of the ion channels in order to compute J
        """
        #C = y#[ aM/μm^3 = m mol/L] aM : atto(10E-18) mol

        D = Simulation._difus_mat(neuron,T,ion,V,t)#[μm^3/ms]

        J = Simulation._flux(neuron,ion,V,S,t)#[aM/ms]

        dCi = Vol1 * (D @ C + J) #[aM/μm^3/ms]
        dCo = k_c @ dCi #dC/dt for connecting nodes

        ### Progressbar
        if not (tq is None):
            n=round(t-t_span[0],3)
            if n>tq.n:
                tq.update(n-tq.n)

        return np.concatenate([dCi,dCo])



    def resample(self,freq):
        self.V=self.V.resample(freq).mean()


    #Caching functions
    @staticmethod
    @lru_cache(128)
    def _difus_mat(neuron,T,ion,V,t):
        """Return the electrodiffusion matrix for :
        - the ion *ion* type sinaps.Ion
        - potential *V* [mV] for each compartment
        - Temperature *T* [K]
        - *t* : index in array V, usually the time

        (we call with V and t as different argument to be cable to
        #use the caching with lru as this method will be used a lot)
        """
        #We
        return neuron.difus_mat(T,ion,V(t))

    @staticmethod
    @lru_cache(128)
    def _flux(neuron,ion,V,S,t):
        """return the transmembrane flux of ion ion (aM/ms attoMol)
                 towards inside

        (we call with V,S and t as different argument to be cable to
        #use the caching with lru as this method will be used a lot)
        """
        #We
        return neuron.J(ion,V(t),S(t),t)
