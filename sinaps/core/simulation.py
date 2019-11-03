# coding: utf-8

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
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


    def __init__(self, neuron, dx=1, atol = 1.49012e-8 ):
        # todo : adapte default absolute tolerance
        self.atol = atol
        self.N = deepcopy(neuron)
        self.idV,idS0 = self.N.init_sim(dx)
        self.idS = idS0 + self.idV[-1] + 1
        self.Cm1 = 1/self.N.capacitance_array()
        self.G = csr_matrix(self.N.conductance_mat())#sparse matrix format for
                                                     #efficiency
        self.k_c = csr_matrix(self.N.connection_mat())

        self.V_S0 = np.concatenate([self.N.V0_array(), self.N.S0_array()])

        self.ions = 0

    def record_ion(self,ion):
        self.ions.append(ion)
        self.V_S0 = np.concatenate([self.V_S0, np.zeros(self.N.nb_comp)])

    def run(self,t_span,method='BDF',**kwargs):
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
                      atol = self.atol,
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




    def ode_function(y,t,idV,idS,Cm,G,k_c,neuron):
        """this function express the ode problem :
        dy/dt = f(y)

        y is a vector containing all state variables of the problem

        V=y[idV] : voltage of each node V=0 at resting potential
                    size n + m where n is the total number of compartiment and
                    m the total number of connecting nodes
        S=y[idS] : other states variables related to the ion channels

        Cm : Capacitance of the membrane for each compartiment size n
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
        dVi = 1/Cm * (G @ V + I) #dV/dt for  compartiment
        dVo = k_c @ dVi #dV/dt for connecting nodes
        dS = neuron.dS(V,S)

        #electrodiffusion equation
        #C = y[idC]TODO
        #dC = (Gk * C) @ V + J + D @ C TODO
        #return np.concatenate([dVi,dVo,dS,dC])
        return np.concatenate([dVi,dVo,dS])

    def resample(self,freq):
        self.V=self.V.resample(freq).mean()
