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
        self.idV,self.idS = self.N.init_sim(dx)
        self.Cm1 = 1/self.N.capacitance_array()[:,np.newaxis]
        self.Vol1 = 1/self.N.volume_array()[:,np.newaxis]
        G = csr_matrix(self.N.conductance_mat())#sparse matrix format for
                                                     #efficiency
        self.k_c = csr_matrix(np.concatenate([np.identity(self.N.nb_comp),
                                       self.N.connection_mat()]))
        self.G = G @ self.k_c
        self.V_S0 = np.zeros(max(np.concatenate((self.N.idV,self.N.idS)))+1)
        self.N.fill_V0_array(self.V_S0)
        self.N.fill_S0_array(self.V_S0)
        self.channels = self.N.all_channels()
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
                                                self.Cm1,self.G,self.channels,
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
        self.V_St=interpolate.interp1d(self.sol.t,self.sol.y,
                                fill_value='extrapolate')

    @staticmethod
    def ode_function(y,t,idV,idS,Cm1,G,channels,tq=None,t_span=None):
        """this function express the ode problem :
        dy/dt = f(y)

        y is a vector containing all state variables of the problem

        V=y[idV] : voltage of each node V=0 at resting potential
                    size n + m where n is the total number of compartiment
        S=y[idS] : other states variables related to the ion channels

        Cm1 : Inverse of the capacitance of the membrane for each compartiment size n
        G : Conductance matrix size n * n

        Voltage equation  for compartiment is given by :
        dV/dt = 1/Cm (G.V + Im )

        neuron : type Neuron contains the information about the behaviour
         of the ion channels in order to compute I
        """
        vectorize = y.ndim >1
        if not vectorize:
            y=y[:,np.newaxis]

        V = y[idV,:]
        dV_S = np.zeros_like(y)
        for c in channels:
            c.fill_I_dS(dV_S,y,t) #current of active ion channels from outisde to inside
        if vectorize:
            dV_S[idV,:] += np.hstack([G @ V[:,k] for k in range(y.shape[1])]) #current inter compartment
        else :
            dV_S[idV,:]  += G @ V
        dV_S[idV,:]  *= Cm1  #dV/dt for  compartiment

        #Progressbar
        if not (tq is None):
            n=round(t-t_span[0],3)
            if n>tq.n:
                tq.update(n-tq.n)

        return dV_S.squeeze()


    def run_diff(self,species=None,temperature=310,method='BDF',atol = 1.49012e-8,**kwargs):
        """Run the simulation diffusion for ion ion
        The simulation for the voltage must have been run before
        """
        tq=tqdm(total=self.t_span[1]-self.t_span[0],unit='ms')

        Simulation._flux.cache_clear()
        Simulation._difus_mat.cache_clear()

        if species is None:
            species = tuple(self.N.species)

        C0 = np.zeros((self.N.nb_comp,len(species)))
        self.N.fill_C0_array(C0,species)

        sol=solve_ivp(lambda t, y:self.ode_diff_function(y,t,species,
                                                temperature,
                                                tq,self.t_span),
                      self.t_span,
                      np.reshape(C0,-1,'F'),
                      method=method,
                      atol = atol,
                      #jac = lambda t, y:self.jac_diff(y,t,ions,temperature),
                       **kwargs)
        tq.close()

        sec,pos = self.N.indexV()
        df = pd.DataFrame(sol.y[:self.N.nb_comp,:].T ,
                          sol.t ,
                          [sec, pos])
        df.columns.names = ['Section','Position (μm)']
        df.index.name='Time'
        self.C = df
        self.sol_diff = sol


    def ode_diff_function(self,y,t,ions,T,tq=None,t_span=None):
        """this function express the ode problem :
        dy/dt = f(y)

        y is a vector containing all state variables of the problem

        C=y : concentration of each node
        [ aM/μm^3 = m mol/L] aM : atto(10E-18) mol
        size n  where n is the total number of compartiment

        V : Voltage array (previously computed)
        S : Status variables of the channels (previously computed)
        T: Temperature (Kelvin)

        Concentration equation  for compartiment is given by :
        dC/dt = 1/Vol(D V + Jc )
        with D : Electrodiffusion matrix size n * n
        Vol : Volume μm^3

        channels: SimuChannel list contains the information about the behaviour
         of the ion channels in order to compute J
        """
        C = np.reshape(y,(self.N.nb_comp,len(ions)),'F')
        dC = np.zeros_like(C)

        if len(ions) > 1:
            dC += np.hstack([self._difus_mat(T,ion,t) @ C[:,k] \
                        for k,ion in enumerate(ions)])
        else:
            dC += self._difus_mat(T,ions[0],t) @ C
        dC += self._flux(ions,t)#[aM/ms]
        dC *= self.Vol1 #[aM/μm^3/ms]

        ### Progressbar
        if not (tq is None):
            n=round(t-t_span[0],3)
            if n>tq.n:
                tq.update(n-tq.n)

        return np.reshape(dC,-1,'F')

    def jac_diff(self,C,t,ions,T):
        D = self._difus_mat(T,ion,t)#[μm^3/ms]
        return (D @ self.k_c).multiply(self.Vol1) #[aM/μm^3/ms]


    def resample(self,freq):
        self.V=self.V.resample(freq).mean()


    #Caching functions
    @lru_cache(128)
    def _difus_mat(self,T,ion,t):
        """Return the electrodiffusion matrix for :
        - the ion *ion* type sinaps.Ion
        - potential *V* [mV] for each compartment
        - Temperature *T* [K]
        - *t* : index in array V, usually the time

        (we call with V and t as different argument to be cable to
        #use the caching with lru as this method will be used a lot)
        """
        #We
        return self.N.difus_mat(T,ion,self.k_c @ self.V_St(t)[self.idV])\
                @ self.k_c

    @lru_cache(128)
    def _flux(self,ions,t):
        """return the transmembrane flux of ion ion (aM/ms attoMol)
                 towards inside

        (we call with V,S and t as different argument to be cable to
        #use the caching with lru as this method will be used a lot)
        """
        J = np.zeros((self.N.nb_comp,len(ions)))
        for c in self.channels:
            c.fill_J(J,ions,self.V_St(t)[:,np.newaxis],t)
        return J
