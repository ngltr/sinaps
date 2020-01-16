# coding: utf-8
from functools import lru_cache
from copy import deepcopy
import warnings

import numpy as np
from numba import njit, jitclass
from numba import int8, float32
from quantiphy import Quantity
import pandas as  pd
from scipy.sparse import dia_matrix
from scipy import interpolate

import sinaps.species as species


PI=np.pi
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


class Neuron:
    """Set of sections connected together.

    Attributes
    ----------
    sections : list(Section)
        `sections` is the list of sections composing the neuron.
    V_ref : float
        Resting potential of the neuron due to long term dynamic of ions channels not modeled
    """

    spatialError = ValueError("""No spatial resolution for the neuron
                    init_sim(dx) must be called before
                    """)

    def __init__(self, V_ref=0):
        """
        V_ref is the resting potential of the neuron (V)
        """

        self.sections=dict()
        self.species=set()
        self.V_ref = V_ref
        self.dx = None
        self._mat=None
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
    if issubclass(type(key), int):
        return list(self.sections)[key]
    elif issubclass(type(key), slice):
        sec = list(self.sections)[key]
    else:
        sec=[s for s in N1.sections if re.match(key,s.name) is not None]
    if len(sec) == 1:
        return sec[0]
    else:
        return SectionList(sec)


    def add_section(self,sec,i,j):
        """Connect nodes
        example : N.add_section(s,i,j)
        With i and j:Int and s:Section
        connect nodes i and j with section s
        """
        if not issubclass(type(sec), Section):
            raise ValueError('Must be a section (sinaps.Section)')
        if hasattr(sec,"N"):
            raise ValueError('This section is already part of a neuron')
        sec.N=self
        sec.i=i
        sec.j=j
        sec.num=len(self.sections)
        self.sections[sec]=(i,j)

    def add_species(self,species,C0={},D={}):
        """Add species to the neuron
        C0 : default value for initial concentration if not defined in the section
        D : default value for difusion coef if not defined in the section

        Add multipe species :
        species is a list
        C0 and D dictionnary with inital value

        """
        if not hasattr(species,"__iter__"):
            if type(C0) in (float,int):
                C0={species:C0}
            if type(D) in (float,int):
                D={species:D}
            species={species}

        self._species = self._species.union(species)
        for sp in species:
            for sec in self:
                if not (sp in sec.C0):
                    if sp in C0:
                        sec.C0[sp] = C0[sp]
                    else:
                        raise InitialConcentrationError(sp)
                if not (sp in sec.D):
                    if sp in D:
                        sec.D[sp] = D[sp]
                    else:
                        raise InitialConcentrationError(sp)#TODO
    @property
    def species(self):
        """`species` is a set of species considered in electrodiffusion simulation."""
        return self._species

    @property
    def nb_nodes(self):
        return max([max(ij) for ij in self.sections.values()])+1

    @property
    def adj_mat(self):
        if self._mat is None:
            n=self.nb_nodes
            self._mat = np.ndarray([n,n],Section)
            for s,ij in self.sections.items():
                self._mat[ij[0],ij[1]]=s
        return self._mat

    def _init_sim(self,dx):
        """Prepare the neuron to run a simulation with spatial resolution dx"""
        self.nb_comp = 0#number of comparment
        self.dx = dx
        for s in self.sections:
            self.nb_comp += s._gen_comp(dx)

        idV0=0
        idS0=self.nb_comp
        for s in self.sections:
            idV0,idS0=s._init_sim(idV0,idS0)

        self.nb_con = max([max(s.i,s.j) for s in self.sections]) + 1
        #number of connecting nodes

        self.idV = np.array(range(idV0),int)
        self.idS = np.array(range(self.nb_comp,idS0),int)

        return self.idV, self.idS

    def _all_channels(self):
        """Return channels objects suitable for the simulation"""
        cch=[]
        for ch_cls in { type(c)  for s in self for c in s.channels}:
            ch = _SimuChannel(ch_cls)
            params={ p:[] for p in ch_cls.param_names}
            idV, idS, surface = [],[],[]
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

    def _capacitance_array(self):
        """Return the menbrane capacitance for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        Cm = np.zeros(self.nb_comp)
        for s in self.sections:
            Cm[s.idV] = s.c_m_array()

        return Cm

    def _volume_array(self):
        """Return the volume of each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        V = np.zeros(self.nb_comp)
        for s in self.sections:
            V[s.idV] = s.volume_array()

        return V

    def _conductance_mat(self):
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

    def _connection_mat(self):
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

    def _fill_V0_array(self,V0):
        """fill the initial potential for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        for s in self.sections:
            s.fill_V0_array(V0)
        V0[self.idV] = V0[self.idV] - self.V_ref #Conversion of reference potential V=0 for the
        #resting potential in the model

    def _fill_S0_array(self,S0):
        """fill initial state variable for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        for s in self.sections:
            s.fill_S0_array(S0)

    def _fill_C0_array(self,C0,ions):
        """Return the initial concentration for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        for s in self.sections:
            s.fill_C0_array(C0,ions)

    #Diffusion functions

    #@lru_cache(1)
    def _difus_array(self,ion):
        """Return the diffusion coefficient D*a/dx between each node for ion
        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError
        cc=np.concatenate
        return cc([ cc([[0],s.difus_array(ion)]) for s in self.sections])

    #@lru_cache(1)
    def _difus_end(self,ion):
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

    def _difus_mat(self,T,ion,Vt):
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
        k=11.6045*species.CHARGE[ion]/T


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

    def indexV(self,species=None):
        """return index for potential vector
        id section
        position
        """
        if species is None:
            return (np.concatenate( [ [s.name]*len(s.idV) for s in self.sections]),
                np.concatenate([s.x for s in self.sections]))
        else:
            return (np.concatenate( [[sp]*self.nb_comp for sp in species]),
            np.concatenate( [ [s.name]*len(s.idV) for s in self.sections]*len(species)),
                np.concatenate([s.x for s in self.sections]*len(species)))

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
        return (np.concatenate( [ [s.name]*len(s.idS) for s in self.sections]),
                np.concatenate([s.index for s in self.sections]))

class Section:
    """This class representss a section of neuron with uniform physical values
    """
    _next_id=0

    def __init__(self, L=100, a=1, C_m=1, R_l=150, V0=0, name=None,
                    C0=None,
                    D=None):
        """
            S=Section(L : length [μm],
                      a : radius [μm],
                      C_m : menbrane capacitance [μF/cm²],
                      R_l : longitudinal resistance [Ohm.cm]
                      [V0]=0 : initial potential (mV)
                      [name]=None,
                      [C0]=None : initial concentration for species (mM/L),
                      [D]=None : diffusion coeficient for species (um2/ms),
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
        if D is None:
            D={}
        self.D = D
        self.channels_c = list()#density channels
        self.channels_p = list()#point channels
        if name is None:
            self.name="Section{:04d}".format(Section._next_id)
        else:
            self.name=name
        Section._next_id +=1


    @property
    def r_l(self):
        """Longitudinal resistance per unit length [GOhm/μm]"""
        return  self.R_l / (PI * self.a**2)

    @property
    def c_m(self):
        """Membrane capacitance per unit length [pF/μm]"""
        return self.C_m * 2 * PI * self.a


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
        + C_m: {} (c_m={})
        + R_l: {} (r_l={})
        + channels: {}
        + point_channels: {}""".format(
            self.name,
            Quantity (self.L*1E-6,'m'),
            Quantity (self.a*1E-6,'m'),
            Quantity (self.C_m*1E-12,'F/μm²'),
            Quantity (self.c_m*1E-12,'F'),
            Quantity (self.R_l*1E9,'Ω.μm'),
            Quantity (self.r_l*1E-12,'Ω'),
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
            warnings.warn(('The section already had a channel of type {} '\
                            .format(type(C))))

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
    def _gen_comp(self,dx):
        """ if dx = 0 , section.dx will be use """
        if dx > 0:
            self.dx=dx
        try:
            self.nb_comp=max(int(np.ceil(self.L/self.dx)),2)
        except AttributeError:
            raise ValueError('You must first define dx as a property for each section before using dx=0')

        return self.nb_comp

    def _init_sim(self,idV0,idS00):
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

    def _continuous(self,param):
        if callable(param):
            return np.vectorize(param)
        elif type(param) is list:
            return interpolate.interp1d([0,self.L],param)
        else:
            return lambda x:param*np.ones_like(x)

    def _param_array(self,param):
        return self._continuous(param)(self.x)

    def _param_array_diff(self,param):
        return self._continuous(param)((self.x[:-1]+self.x[1:])/2)

    def _param_array_end(self,param):
        return self._continuous(param)([self.x[0]/2,(self.L + self.x[-1])/2])

    def _fill_V0_array(self,V0):
        """fill the initial potential for each nodes
        init_sim(dx) must have been previously called
        """
        V0[self.idV] = self._param_array(self.V0) #[mV]

    def _fill_S0_array(self,S0):
        """fill the initial state variables values for each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        for c in self.channels:
            s0 = c.__S0(len(self.x))
            for k in range(c.nb_var):
                S0[c.idS[k]] = s0[k]

    def _fill_C0_array(self,C0,ions):
        """Return the initial concentration for each nodes
        init_sim(dx) must have been previously called
        """
        for k,ion in enumerate(ions):
            if ion in self.C0:
                C0[self.idV,k] = self._param_array(self.C0[ion]) #[mM/L]
            else:
                raise InitialConcentrationError(ion)


    def _c_m_array(self):
        """Return the membrane capacitance for each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        return self._surface_array() * self._param_array(self.C_m) #[nF]

    def _volume_array(self):
        """Return the volume of each compartment
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        return np.diff(self.xb) * self._param_array(self.a)**2 * PI #[um3]

    def _surface_array(self):
        """Return the menbrane surface of each compartment
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        return np.diff(self.xb) * self._param_array(self.a) * 2 * PI #[um2]

    def _r_l_array(self):
        """Return the longitunal resistance between each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        return np.diff(self.x) *  self._param_array_diff(self.R_l) \
                /(self._param_array_diff(self.a)**2 *PI)#[MΩ]

    def _r_l_end(self):
        """Return the longitunal resistance between the start/end of the section
        and the first/last node
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        return np.array([self.x[0],(self.L - self.x[-1])]) \
                *  self._param_array_end(self.r_l)  #[MΩ]

    #Diffusion functions
    def _difus_array(self,ion):
        """Return the diffusion coefficient D*a/dx between each node for ion
        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        if ion in self.D:
            D=self.D[ion]#[μm^2/ms]
        else:
             D=species.DIFFUSION_COEF[ion]
        return self._param_array_diff(D) * (PI * self._param_array_diff(self.a)**2)\
                / np.diff(self.x)  #[μm^3/ms]

    def _difus_end(self,ion):
        """Return the diffusion coefficient D*a/dx betweenthe start/end of the section
        and the first/last node for ion
        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        if ion in self.D:
            D=self.D[ion]#[μm^2/ms]
        else:
             D=species.DIFFUSION_COEF[ion]
        return self._param_array_end(D)  \
                *(PI * self._param_array_end(self.a)**2) \
                / np.array([self.x[0],(self.L - self.x[-1])])  #[μm^3/ms]


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
    def _dS(V,*S,t,**unused):
        return (0,) * len(S)

    def S0(self):
        return (0,) * self.nb_var

    def _Section__S0(self,n):
        if self.nb_var > 1:
            return np.concatenate([[S] *n for S in self.S0()])
        else:
            return [self.S0()] * n

    def density_str(self):
        if hasattr(self,'position'):
            return ''
        else:
            return '/μm²'


class _SimuChannel:
    """Class for vectorization of channels used for efficiency"""
    def __init__(self,ch_cls):
        self.I=njit(ch_cls._I)
        self.dS=njit(ch_cls._dS)
        if hasattr(ch_cls,'_J'):
            self.J=njit(ch_cls._J)
        self.__name__=(ch_cls.__name__)

    def fill_I_dS(self,y,V_S,t):
        """fill the array y with :
        - The transmembrane curent of the channels self
        - the variation of the states variable of the channels self
        V_S : current values for potential and state variables
        t : current time (ms)
        """
        V=V_S[self.idV,:]
        S=[V_S[self.idS[k],:] for k in range(self.nb_var)]
        np.add.at(y,np.s_[self.idV,:],self.I(V,*S,t,**self.params) * self.k)
        if self.nb_var:
            dS = self.dS(V,*S,t,**self.params)
            if self.nb_var >1:
                for k in range(self.nb_var):
                    y[self.idS[k],:] = dS[k]
            else:
                y[self.idS[0],:] = dS

    def fill_J(self,y,ions,V_S,t):
        """fill the array y with :
            - The flux of ion of the channels self
            V_S : current values for potential and state variables
            t : current time (ms)"""
        #continuous channels
        if hasattr(self,'J'):
            V=V_S[self.idV]
            S=[V_S[self.idS[k]] for k in range(self.nb_var)]
            for k,ion in enumerate(ions):
                np.add.at(y,np.s_[self.idV,k],(self.J(ion,V,*S,t,**self.params)
                                              * self.k).squeeze())


@jitclass({'charge':int8,'D':float32})
class _Species:
    """This class represent a species"""
    def __init__(self,charge,D):
        """
        charge : int electric charge of the ion (ex 2 for Ca2+ and -1 for Cl-)
        D : diffusion coef of the ion um2/ms
        """
        self.charge = charge
        self.D = D
