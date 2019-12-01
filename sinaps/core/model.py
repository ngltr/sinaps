# coding: utf-8
from functools import lru_cache
from copy import deepcopy

import numpy as np
from numba import njit
pi=np.pi
from quantiphy import Quantity
import pandas as  pd
from scipy.sparse import dia_matrix
from scipy import interpolate
"""
Units :

Voltage : mV (E-3)
Time : ms (E-3)
Distance : μm (E-6)
Resistance : GΩ (E9)
Conductance : nS (E-9)
Capacitance : pF (E-12)
Current : pA (E-12)
Mol :aM (E-18)

"""
class InitialConcentrationError(ValueError):
    def __init__(self, ion):
        super().__init__(
            """No initial concentration defined for the ion {}
               define it using [section].C0=dict([ion]=[initial_concentration])
                    """.format(ion))


class Section:
    """This class representss a section of neuron with uniform physical values
    """


    def __init__(self, L=100, a=1, C_m=1, R_l=150, V0=0, name=None, C0=None):
        """
            S=Section(L : length [μm],
                      a : radius [μm],
                      C_m : menbrane capacitance [μF/cm²],
                      R_l : longitudinal resistance [Ohm.cm]
                      [V0]=0 : initial potential (mV)
                      [name],
                      [C0]={} : initial concentration for ions (mM/L),
                      )

            resistance and capacitance given per membrane-area unit
        """
        self.L = L
        self.a = a
        self.C_m = C_m/100  # conversion to pF/μm2: 1 μF/cm2 = 0.01 pF/μm2
        self.R_l = R_l/1E5 # conversion to GΩ.μm: 1 Ω.cm = E-5 GΩ.μm
        self.V0 = V0
        if C0 is None:
            C0={}
        self.C0 = C0
        self.channels_c = list()#density channels
        self.channels_p = list()#point channels
        if name is None:
            self.name=id(self)
        else:
            self.name=name
        self.D=dict()


    @property
    def r_l(self):
        """longitudinal resistance per unit length [GOhm/μm]"""
        return  self.R_l / (pi * self.a**2)

    @property
    def c_m(self):
        """membrane capacitance per unit length [pF/μm]"""
        return self.C_m * 2 * pi * self.a


    def __repr__(self):
        return """Section(name={}, L={}, a={}, C_m={}, R_l={}, channels : {}, point_channels : {})""".format(
            self.name,
            Quantity (self.L*1E-6,'m'),
            Quantity (self.a*1E-6,'m'),
            Quantity (self.C_m*1E-4,'F/cm²'),
            Quantity (self.R_l*1E5,'Ω.cm'),
            [c for c in self.channels_c],
            ['{}:{}'.format(c.position,c)for c in self.channels_p])
    def __str__(self):
        return "Section {}".format(self.name)

    def _repr_markdown_(self):
        return """Section **{}**
+ L: {}
+ a: {}
+ C_m: {}
+ R_l: {}
+ channels: {}
+ point_channels: {}""".format(
            self.name,
            Quantity (self.L*1E-6,'m'),
            Quantity (self.a*1E-6,'m'),
            Quantity (self.C_m*1E-12,'F/μm²'),
            Quantity (self.R_l*1E9,'Ω.μm'),
            '\n  + '.join([str(c) for c in self.channels_c]),
            '\n  + '.join(['{}:{}'.format(c.position,c)for c in self.channels_p])
            )


    def __copy__(self):
        return Section(self.name)


    def add_channel(self,C,x=None):
        """Add channel to the section
        C: type channel
        position : relative position along the section (between 0 and 1),
        if None (default) the channels is treated as a continuous density channel
        giving a surfacic current in nA/μm2
        """
        #TODO copy the objet to avoid sharing same
        #betwenn several sections

        if not issubclass(type(C), Channel):
            raise ValueError('Must be a channel (sinaps.Channel)')

        if type(C) in [type(c) for c in self.channels]:
            raise ValueError('Only one channel of type {} per section is authorized'\
                            .format(type(C)))

        C=deepcopy(C)
        if x is None:
            self.channels_c.append(C)
        else:
            C.position=x
            self.channels_p.append(C)

    @property
    def channels(self):
        return self.channels_c + self.channels_p
    ## Initilialisation functions

    def gen_comp(self,dx):
        """ if dx = 0 , sec.dx will be use """
        if dx > 0:
            self.dx=dx
        try:
            self.nb_comp=max(int(np.ceil(self.L/self.dx)),2)
        except AttributeError:
            raise ValueError('You must first define dx as a property for each section before using dx=0')

        return self.nb_comp

    def init_sim(self,idV0,idS00):
        """Prepare the section to run a simulation with spatial resolution dx
        gen_comp must have been run before
        idV0 : first indices for Voltage
        idS00 : first indice for state variables
        """


        n = self.nb_comp
        self.dx = self.L/n
        self.x = np.linspace(self.dx/2,self.L-self.dx/2,n)##center of the compartiment, size n
        self.xb = np.linspace(0,self.L,n+1)##border of the compartiment, size n+1
        self.idV = idV0 + np.array(range(n),int)
        idS0=idS00
        for c in self.channels_c:
            c.idS = [k*n + idS0 + np.array(range(n),int) for k in range(c.nb_var)]
            c.idV = self.idV
            idS0 +=  n * c.nb_var# * state variables
        for c in self.channels_p:
            c.idV = idV0 + int(min(c.position * n, n-1))#index of the compartment related to the point process
            c.idS = [ k + idS0 + np.array([0],int) for k in range(c.nb_var)]
            idS0 += c.nb_var # * state variables
        self.nb_var=idS0-idS00
        return idV0+n,idS0


    def continuous(self,param):
        if callable(param):
            return np.vectorize(param)
        elif type(param) is list:
            return interpolate.interp1d([0,self.L],param)
        else:
            return lambda x:param*np.ones_like(x)

    def param_array(self,param):
        return self.continuous(param)(self.x)

    def param_array_diff(self,param):
        return self.continuous(param)((self.x[:-1]+self.x[1:])/2)

    def param_array_end(self,param):
        return self.continuous(param)([self.x[0]/2,(self.L + self.x[-1])/2])

    def fill_V0_array(self,V0):
        """fill the initial potential for each nodes
        init_sim(dx) must have been previously called
        """
        V0[self.idV] = self.param_array(self.V0) #[mV]

    def fill_S0_array(self,S0):
        """fill the initial state variables values for each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        for c in self.channels:
            s0 = c.__S0(len(self.x))
            for k in range(c.nb_var):
                S0[c.idS[k]] = s0[k]

    def fill_C0_array(self,ion,C0):
        """Return the initial concentration for each nodes
        init_sim(dx) must have been previously called
        """
        if ion in self.C0:
            C0[self.idV] = self.param_array(self.C0[ion]) #[mM/L]
        else:
            raise InitialConcentrationError(ion)


    def c_m_array(self):
        """Return the membrane capacitance for each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        return self.surface_array() * self.param_array(self.C_m) #[nF]

    def volume_array(self):
        """Return the volume of each compartment
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        return np.diff(self.xb) * self.param_array(self.a)**2 * pi #[um3]

    def surface_array(self):
        """Return the menbrane surface of each compartment
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        return np.diff(self.xb) * self.param_array(self.a) * 2 * pi #[um2]


    def r_l_array(self):
        """Return the longitunal resistance between each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        return np.diff(self.x) *  self.param_array_diff(self.R_l) \
                /(self.param_array_diff(self.a)**2 *pi)#[MΩ]

    def r_l_end(self):
        """Return the longitunal resistance between the start/end of the section
        and the first/last node
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        return np.array([self.x[0],(self.L - self.x[-1])]) \
                *  self.param_array_end(self.r_l)  #[MΩ]

    #Diffusion functions

    def difus_array(self,ion):
        """Return the diffusion coefficient D*a/dx between each node for ion
        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        if ion in self.D:
            D=self.D[ion]#[μm^2/ms]
        else:
             D=ion.D
        return self.param_array_diff(D) * (pi * self.param_array_diff(self.a)**2)\
                / np.diff(self.x)  #[μm^3/ms]

    def difus_end(self,ion):
        """Return the diffusion coefficient D*a/dx betweenthe start/end of the section
        and the first/last node for ion
        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        if ion in self.D:
            D=self.D[ion]#[μm^2/ms]
        else:
             D=ion.D
        return self.param_array_end(D)  \
                *(pi * self.param_array_end(self.a)**2) \
                / np.array([self.x[0],(self.L - self.x[-1])])  #[μm^3/ms]

    ## Simulation functions


    def I(self,V,S,t):
        """return the transmembrane current"""
        I= np.zeros(V.shape)
        #continuous channels
        for c in self.channels_c:
            I[self.idV] += c.__I(V,S,t) * self.dx * 2 * pi * self.a
            #todo taking into account variable radius

        #point channels
        for c in self.channels_p:
            I[self.idV] += c.__I(V,S,t)

        return I


    def dS(self,V,S):
        """return the differential of the state variables of the ion channels"""
        dS = np.zeros(S.shape)
        for channels in (self.channels_c,self.channels_p):
            for c in channels:
                if len(c['idS']):
                    dS[c['idS']] = c['obj'].__dS(V,S[c['idS']])
        return dS

    def J(self,ion,V,S,t):
        """return the transmembrane flux of ion ion (aM/ms attoMol)
         towards inside"""
        J = np.zeros(V.shape)
        #continuous channels
        for c in self.channels_c:
            J += c['obj'].__J(ion,V,S[c['idS']],t)\
                 * self.dx * 2 * pi * self.a
            #todo taking into account variable radius

        #point channels
        for c in self.channels_p:
            J[c['idV']] += c['obj'].__J(ion,V[c['idV']],S[c['idS']],t)

        return J


    ## Save results functions

    def save_result(self,V,S):
        self.V = V
        for c in self.channels_c + self.channels_p:
            if len(c['idS']):
                c['obj'].save_result(S[c['idS']])



class Channel:
    """
    Generic class describing a channel or a group of channel
    To implement a channel C, it is necessary to implement  :
     - a function C.I(V,*st_vars) returning the net  current towards inside
      ([A] for point channel and [pA/μm2] for density channels ) of the mechanism with
     V the voltage (:array) and *st_vars the state variable of the mechanism (:array)
      - If there is state variables a function C.dS(V,*st_vars) returning a
      tuple with the differential of each state variable
       and a function S0 returning a tuple with the initial value for each state
       variable
    """
    nb_var = 0
    param_names=()
    params = {}

    def I(self,V,*S,t):
        return self._I(V,*S,t=t,**self.params)

    def J(self,V,*S,t):
        return self._I(V,*S,t=t,**self.params)

    def dS(self,V,*S,t=0):
        return self._I(V,*S,t=t,**self.params)

    @staticmethod
    def _I(V,t):
        return 0 * V
    @staticmethod
    def _J(ion,V,t):
        return 0 * V
    @staticmethod
    def _dS(V,*S,t,**unused):
        return (0,) * len(S)

    def S0(self):
        return (0,) * self.nb_var


    def _Section__S0(self,n):
        if hasattr(self,'position'):
            n=1
        try:
            return [[S] * n for S in self.S0()]
        except TypeError: # S0 is not iterable
            return [[self.S0()] * n]

    def _Section__I(self,V,S,t):
        if self.nb_var > 1:
            n=len(V)
            st_vars=[S[k*n:k*n+n] for k in range(self.nb_var)]
            return self._I(V,*st_vars,t)
        else:
            return self._I(V,S,t=t)

    def _Section__J(self,ion,V,S,t):
        if self.nb_var > 1:
            n=len(V)
            st_vars=[S[k*n:k*n+n] for k in range(self.nb_var)]
            return self.J(ion,V,*st_vars,t)
        else:
            return self.J(ion,V,S,t=t)

    def _Section__dS(self,V,S):
        if self.nb_var > 1:
            n=len(V)
            st_vars=[S[k*n:k*n+n] for k in range(self.nb_var)]
            return np.concatenate(self.dS(V,*st_vars))
        else:
            return self.dS(V,S)


    def density_str(self):
        if hasattr(self,'position'):
            return ''
        else:
            return '/μm²'


class Neuron:
    """This class represents a neuron
    N=Neuron(sections)


    """

    spatialError = ValueError("""No spatial resolution for the neuron
                    init_sim(dx) must be called before
                    """)


    def __init__(self, V_ref=0):
        """
        V_ref is the resting potential of the neuron (V)
        """

        self.sections=list()
        self.V_ref = V_ref
        self.dx = None
        self.mat=None
        self._x=None
        self._y=None

    def __repr__(self):
        return "Neuron({})".format(
            ['{}-{}: {}'.format(s.i,s.j,s.__repr__())
                for s in self.sections]
                )
    def __len__(self):
        return len(self.sections)

    def __getitem__(self,key):
        return self.sections[key]

    def __setitem__(self,key,value):
        self.sections[key]=value

    def add_section(self,s,i,j):
        """Connect nodes
        example : N.add_section(s,i,j)
        With i and j:Int and s:Section
        connect nodes i and j with section s
        """
        if not issubclass(type(s), Section):
            raise ValueError('Must be a section (sinaps.Section)')
        sec=deepcopy(s)
        sec.i=i
        sec.j=j
        sec.num=len(self.sections)
        self.sections.append(sec)

    @property
    def nb_nodes(self):
        return max([max(s.i,s.j) for s in self.sections])+1

    @property
    def adj_mat(self):
        if self.mat is None:
            n=self.nb_nodes
            self.mat = np.ndarray([n,n],Section)
            for s in self.sections:
                self.mat[s.i,s.j]=s
        return self.mat

    def init_sim(self,dx):
        """Prepare the neuron to run a simulation with spatial resolution dx"""
        self.nb_comp = 0#number of comparment
        self.dx = dx
        for s in self.sections:
            self.nb_comp += s.gen_comp(dx)

        idV0=0
        idS0=self.nb_comp
        for s in self.sections:
            idV0,idS0=s.init_sim(idV0,idS0)

        self.nb_con = max([max(s.i,s.j) for s in self.sections]) + 1
        #number of connecting nodes

        self.idV = np.array(range(idV0),int)
        self.idS = np.array(range(self.nb_comp,idS0),int)

        return self.idV, self.idS

    def all_channels(self):
        """Return channels objects suitable for the simulation"""
        cch=[]
        for ch_cls in { type(c)  for s in self for c in s.channels}:
            ch = SimuChannel(ch_cls)
            params={ p:[] for p in ch_cls.param_names}
            idV = []
            idS = []
            surface = []
            for s in self:
                for c in s.channels:
                    if type(c) is ch_cls:
                        for p in params:
                            if hasattr(c,'position'):
                                value=c.params[p]
                            else:
                                value=s.param_array(c.params[p])
                            params[p].append(value)
                        idV.append(c.idV)
                        idS.append(c.idS)
                        if hasattr(c,'position'):
                            surface.append([1])
                        else:
                            surface.append(s.surface_array())
            ch.params = {k:np.hstack(v)[:,np.newaxis] for k,v in params.items()}
            ch.idV = np.hstack(idV)
            ch.nb_var = ch_cls.nb_var
            if ch.nb_var:
                ch.idS = [np.hstack(idS)[k,:] for k in range(ch.nb_var)]
            else:
                ch.idS=[]
            ch.k= np.concatenate(surface)[:,np.newaxis]
            cch.append(ch)
        return cch

    def capacitance_array(self):
        """Return the menbrane capacitance for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        Cm = np.zeros(self.nb_comp)
        for s in self.sections:
            Cm[s.idV] = s.c_m_array()

        return Cm

    def volume_array(self):
        """Return the volume of each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        V = np.zeros(self.nb_comp)
        for s in self.sections:
            V[s.idV] = s.volume_array()

        return V

    def conductance_mat(self):
        """Return conductance matrix G of the neuron
        G is a sparse matrix
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        n = self.nb_comp
        m = self.nb_con

        #Initilialisation conductance matrix size n * n+m
        G = np.zeros([n, n + m])

        for s in self.sections:
            # longitudinal conductance in a section
            g_l = 1/s.r_l_array()
            G[s.idV[:-1],s.idV[1:]] = + g_l
            G[s.idV[1:],s.idV[:-1]] = + g_l
            G[s.idV[:-1],s.idV[:-1]] += - g_l
            G[s.idV[1:],s.idV[1:]] +=  - g_l
            #  conductance between connecting nodes and first/last compartment
            g_end = 1/s.r_l_end()
            ends = [0,-1]
            G[s.idV[ends],[s.i+n,s.j+n]] =  + g_end
            G[s.idV[ends],s.idV[ends]] +=  - g_end

        return G

    def connection_mat(self):
        """Return connection matrix k of the neuron
        giving the relation between the voltage of the connecting nodes and the
        voltage of the compartiment nodes
        init_sim(dx) must have been previously called

        V0 = sum (gk.Vk) / sum(gk)

        """
        if self.dx is None:
            raise Neuron.spatialError

        #Initilialisation connection matrix size m * n
        k = np.zeros([self.nb_con, self.nb_comp])

        for s in self.sections:
            g_end = 1/s.r_l_end()
            # conductance of leak channels
            k[[s.i,s.j],s.idV[[0,-1]]] = g_end
        k = k/k.sum(axis=1,keepdims=True)
        return k


    def fill_V0_array(self,V0):
        """fill the initial potential for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        for s in self.sections:
            s.fill_V0_array(V0)
        V0[self.idV] = V0[self.idV] - self.V_ref #Conversion of reference potential V=0 for the
        #resting potential in the model

    def fill_S0_array(self,S0):
        """fill initial state variable for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        for s in self.sections:
            s.fill_S0_array(S0)


    def C0_array(self,ion):
        """Return the initial concentration for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        C0 = np.zeros_like(self.idV,float)
        for s in self.sections:
            s.fill_C0_array(ion,C0)
        return C0

    #Diffusion functions

    #@lru_cache(1)
    def difus_array(self,ion):
        """Return the diffusion coefficient D*a/dx between each node for ion
        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError
        cc=np.concatenate
        return cc([ cc([[0],s.difus_array(ion)]) for s in self.sections])

    #@lru_cache(1)
    def difus_end(self,ion):
        """Return the diffusion coefficient D*a/dx betweenthe start/end of the section
        and the first/last node for ion
        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError
        return np.array([s.difus_end(ion) for s in self.sections])\
                    .swapaxes(0,1).flatten()
                    #return the firs node of all section first
                    #then the last node of all sections


    def difus_mat(self,T,ion,Vt):
        """Return the electrodiffusion matrix for :
        - the ion *ion* type sinaps.Ion
        - potential *V* [mV] for each compartment
        - Temperature *T* [K]

        init_sim(dx) must have been previously called
        """
        cc=np.concatenate
        if self.dx is None:
            raise Neuron.spatialError
        n = self.nb_comp
        m = self.nb_con

        Vtt=Vt[1:]-Vt[:-1]

        #e=1.6021766208e-19 A.s electron charge
        #e=1.6021766208e-4 pA.ms
        #kb=1.38064852e-23 J.K-1 = Ω.A^2.s Boltzman constant
        #kb=1.38064852e-05 GΩ.pA^2.ms
        #e/kb=11.6045 1/(pA.GΩ)=mV^-1
        #k=ez/kbT
        k=11.6045*ion.charge/T


        #flux inside a section
        d=np.resize(self.difus_array(ion),n+m)#[μm^3/ms]
        j=dia_matrix((
        np.array([cc([[0],d[1:]*(1+k/2*Vtt)]),
              cc([d[1:]*(-1+k/2*Vtt),[0]
                 ])]),
                 [0,-1]),shape=(n+1,n+m)).tolil()
        j[-1]=0 # No flux for last compartiment
        jtt=(j[1:]-j[:-1]).tolil() # flux difference [μm^3/ms]

        #flux for connecting nodes
        d_end=self.difus_end(ion)
        idA=cc([[s.idV[0] for s in self.sections],
                [s.idV[-1] for s in self.sections]])
        idB=cc([[s.i+n for s in self.sections],
                [s.j+n for s in self.sections]])
        Vttk=k/2*(Vt[idA] - Vt[idB])
        jtt[idA,idB] =  d_end*(1-Vttk)
        jtt[idA,idA] +=  d_end*(-1-Vttk)
        return jtt #[μm^3/ms]

    def I(self,V,S,t):
        """return the transmembrane current towards inside"""
        I = np.zeros(self.nb_comp)
        for s in self.sections:
            I[s.idV] = s.I( V[s.idV], S,t)
        return I



    def dS(self,V,S):
        """return the differential of the state variables of the ion channels"""
        dS = np.zeros_like(S)
        for s in self.sections:
            dS[s.idS] = s.dS(V[s.idV] + self.V_ref, S[s.idS])
        return dS


    def J(self,ion,V,S,t):
        """return the transmembrane flux of ion ion (aM/ms attoMol)
         towards inside"""
        J = np.zeros(self.nb_comp)
        for s in self.sections:
            J[s.idV] = s.J(ion, V[s.idV], S[s.idS],t)
        return J



    def indexV(self):
        """return index for potential vector
        id section
        position
        """
        return (np.concatenate( [ [s.num]*len(s.idV) for s in self.sections]),
                np.concatenate([s.x for s in self.sections]))

    def indexV_flat(self):
        """return position (flattened) for potential vector :"""
        x=0
        ind=[]
        for s in self.sections:
            ind=np.append(ind,s.x+x)
            x += s.L
        return ind

    def indexS(self):
        """return index for state variables
        id section
        channel
        variable
        position
        """
        #todo
        return (np.concatenate( [ [s.num]*len(s.idS) for s in self.sections]),
                np.concatenate([s.index for s in self.sections]))

class SimuChannel:
    """Class for vectorization of channels used for efficiency"""
    def __init__(self,ch_cls):
        self.I=njit(ch_cls._I)
        self.dS=njit(ch_cls._dS)
        self.__name__=(ch_cls.__name__)

    def fill_I_dS(self,y,V_S,t):
        V=V_S[self.idV,:]
        S=[V_S[self.idS[k],:] for k in range(self.nb_var)]
        y[self.idV,:] += self.I(V,*S,t,**self.params) * self.k
        if self.nb_var:
            dS = self.dS(V,*S,t,**self.params)
            if self.nb_var >1:
                for k in range(self.nb_var):
                    y[self.idS[k],:] = dS[k]
            else:
                y[self.idS[0],:] = dS



class Ion:
    """This class represent an ionic specie"""
    def __init__(self,name,charge,D):
        """
        name : str
        charge : int electric charge of the ion (ex 2 for Ca2+ and -1 for Cl-)
        D : diffusion coef of the ion um2/ms
        """
        self.name = name
        self.charge = charge
        self.D = D

    def __repr__(self):
        return "{}{}{}".format(self.name,
                                abs(self.charge) if self.charge>1 else "",
                                "+" if self.charge >0 else "-"
                                )
