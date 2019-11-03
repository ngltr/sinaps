# coding: utf-8
import numpy as np
pi=np.pi
from quantiphy import Quantity
import pandas as  pd


"""
Units :

Voltage : mV
Time : ms
Distance : μm
Resistance : GΩ
Conductance : nS
Capacitance : pF
Current : pA

"""



class Section:
    """This class representss a section of neuron with uniform physical values
    """
    def __init__(self, L=100, a=1, C_m=1, R_l=150, V0=0, name=None):
        """
            S=Section(L : length [μm],
                      a : radius [μm],
                      C_m : menbrane capacitance [μF/cm²],
                      R_l : longitudinal resistance [Ohm.cm]
                      [V0]=0 : initial potential (mV)
                      [name],
                      )

            resistance and capacitance given per membrane-area unit
        """
        self.L = L
        self.a = a
        self.C_m = C_m/100  # conversion to pF/μm2: 1 μF/cm2 = 0.01 pF/μm2
        self.R_l = R_l/1E5 # conversion to GΩ.μm: 1 Ω.cm = E-5 GΩ.μm
        self.V0 = V0
        self.channels_c = list()#density channels
        self.channels_p = list()#point channels
        if name is None:
            self.name=id(self)
        else:
            self.name=name


    @property
    def r_l(self):
        """longitudinal resistance per unit length [GOhm/μm]"""
        return  self.R_l /  pi / self.a**2

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
            [c['obj'] for c in self.channels_c],
            ['{}:{}'.format(c['pos'],c['obj'])for c in self.channels_p])
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
            '\n  + '.join([str(c['obj']) for c in self.channels_c]),
            '\n  + '.join(['{}:{}'.format(c['pos'],c['obj'])for c in self.channels_p])
            )


    def __copy__(self):
        return Section(self.name)


    def add_channel_c(self,C):
        """Add density channels to the section
        C: type channel
        the channels are continuous density giving a surfacic current in nA/μm2
        """
        self.channels_c.append({'obj':C})

    def add_channel_p(self,C,x):
        """Add point channels to the section
        C: type channel
        x: type float, is the relative position of the channel in the section
        the channels are point process giving a current in nA
        """
        self.channels_p.append({'obj':C,'pos':x})



    ## Initilialisation functions

    def init_sim(self,dx):
        """Prepare the section to run a simulation with spatial resolution dx
        if dx = 0 , sec.dx will be use

        """
        idS0=0
        if dx > 0:
            self.dx=dx
        try:
            n=max(int(np.ceil(self.L/self.dx)),2)
        except AttributeError:
            raise ValueError('You must first define dx as a property for each section before using dx=0')
        self.dx=self.L/n
        self.x=np.linspace(self.dx/2,self.L-self.dx/2,n)##center of the compartiment, size n
        self.xb=np.linspace(0,self.L,n+1)##border of the compartiment, size n+1
        self.idV=np.array(range(n),int)
        for c in self.channels_c:
            nv = c['obj'].nb_var * n
            c['idS']=idS0 + np.array(range(nv),int)
            idS0 +=  nv # * state variables
        for c in self.channels_p:
            c['idV']=min(c['pos'] * n, n-1)#index of the compartment related to the point process
            c['idS']=idS0 + np.array(range(c['obj'].nb_var),int)
            idS0 += c['obj'].nb_var # * state variables

        self.idS = np.array(range(idS0),int)
        return self.idV,self.idS

    def V0_array(self):
        """Return the initial potential for each nodes
        init_sim(dx) must have been previously called
        """
        return np.ones_like(self.x) * self.V0 #[mV]

    def S0_array(self):
        """Return the resting potential for each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        S0=np.zeros_like(self.idS,float)
        for c in self.channels_c + self.channels_p:
            if len(c['idS']):
                S0[c['idS']] = c['obj'].__S0(len(self.x))
        return S0

    def c_m_array(self):
        """Return the membrane capacitance for each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        return np.diff(self.xb) * self.c_m #[nF]



    def r_l_array(self):
        """Return the longitunal resistance between each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        return np.diff(self.x) *  self.r_l  #[MΩ]

    def r_l_end(self):
        """Return the longitunal resistance between the start/end of the section
        and the first/last node
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        return np.array([self.x[0],(self.L - self.x[-1])]) *  self.r_l  #[MΩ]



    ## Simulation functions


    def I(self,V,S,t):
        """return the transmembrane current"""
        I= np.zeros(V.shape)
        #continuous channels
        for c in self.channels_c:
            I += c['obj'].__I(V,S[c['idS']],t) * self.dx * 2 * pi * self.a
            #todo taking into account variable radius

        #point channels
        for c in self.channels_p:
            I[c['idV']] += c['obj'].__I(V[c['idV']],S[c['idS']],t)

        return I


    def dS(self,V,S):
        """return the differential of the state variables of the ion channels"""
        dS = np.zeros(S.shape)
        for channels in (self.channels_c,self.channels_p):
            for c in channels:
                if len(c['idS']):
                    dS[c['idS']] = c['obj'].__dS(V,S[c['idS']])
        return dS



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

    def __init__(self):
        self.nb_var = 0

    def I(self,V,*st_vars,t):
        return 0 * V

    def dS(self,V,*st_vars):
        return [0] * self.nb_var

    def S0(self):
        return [0] * self.nb_var


    def _Section__S0(self,n):
        if self.nb_var > 1:
            return np.concatenate([[S] *n for S in self.S0()])
        else:
            return self.S0()


    def _Section__I(self,V,S,t):
        if self.nb_var > 1:
            n=len(V)
            st_vars=[S[k*n:k*n+n] for k in range(self.nb_var)]
            return self.I(V,*st_vars,t)
        else:
            return self.I(V,S,t)

    def _Section__dS(self,V,S):
        if self.nb_var > 1:
            n=len(V)
            st_vars=[S[k*n:k*n+n] for k in range(self.nb_var)]
            return np.concatenate(self.dS(V,*st_vars))
        else:
            return self.dS(V,S)


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
            ['{}-{}: {}'.format(s['i'],s['j'],s['obj'].__repr__())
                for s in self.sections]
                )
    def __len__(self):
        return len(self.sections)

    def __getitem__(self,key):
        return self.sections[key]['obj']

    def __setitem__(self,key,value):
        self.sections[key]['obj']=value

    def add_section(self,s,i,j):
        """Connect nodes
        example : N.add_section(s,i,j)
        With i and j:Int and s:Section
        connect nodes i and j with section s
        """
        self.sections.append({'i':i,'j':j,'obj':s,'num':len(self.sections)})

    @property
    def nb_nodes(self):
        return max([max(s['i'],s['j']) for s in self.sections])+1

    @property
    def adj_mat(self):
        if self.mat is None:
            n=self.nb_nodes
            self.mat = np.ndarray([n,n],Section)
            for s in self.sections:
                self.mat[s['i'],s['j']]=s['obj']
        return self.mat

    def init_sim(self,dx):
        """Prepare the neuron to run a simulation with spatial resolution dx"""
        idV0=idS0=0
        self.dx = dx
        for s in self.sections:
            idV,idS=s['obj'].init_sim(dx)
            s['idV'] = idV0+idV
            s['idS'] = idS0+idS
            idV0 += len(idV)
            idS0 += len(idS)


        self.nb_comp = idV0 # total number of compartment
        self.nb_con = max([max(s['i'],s['j']) for s in self.sections]) + 1
        #number of connecting nodes

        self.idV = np.array(range(idV0 + self.nb_con),int)
        self.idS = np.array(range(idS0),int)
        return self.idV, self.idS


    def capacitance_array(self):
        """Return the menbrane capacitance for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        Cm = np.zeros(self.nb_comp)
        for s in self.sections:
            Cm[s['idV']] = s['obj'].c_m_array()

        return Cm

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
            g_l = 1/s['obj'].r_l_array()
            G[s['idV'][:-1],s['idV'][1:]] = + g_l
            G[s['idV'][1:],s['idV'][:-1]] = + g_l
            G[s['idV'][:-1],s['idV'][:-1]] += - g_l
            G[s['idV'][1:],s['idV'][1:]] +=  - g_l
            #  conductance between connecting nodes and first/last compartment
            g_end = 1/s['obj'].r_l_end()
            ends = [0,-1]
            G[s['idV'][ends],[s['i']+n,s['j']+n]] =  + g_end
            G[s['idV'][ends],s['idV'][ends]] +=  - g_end

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
            g_end = 1/s['obj'].r_l_end()
            # conductance of leak channels
            k[[s['i'],s['j']],s['idV'][[0,-1]]] = g_end
        k = k/k.sum(axis=1,keepdims=True)
        return k


    def V0_array(self):
        """Return the initial potential for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        V0 = np.zeros_like(self.idV,float)
        for s in self.sections:
            V0[s['idV']]=s['obj'].V0_array()
        V0[-self.nb_con:] = self.connection_mat() @ V0[:-self.nb_con]
        return V0 - self.V_ref #Conversion of reference potential V=0 for the
        #resting potential in the model

    def S0_array(self):
        """Return initial stae variable for each nodes
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise Neuron.spatialError

        S0=np.zeros_like(self.idS,float)
        for s in self.sections:
            S0[s['idS']]=s['obj'].S0_array()
        return S0



    def I(self,V,S,t):
        """return the transmembrane current towards inside"""
        I= np.zeros(self.nb_comp)
        for s in self.sections:
            I[s['idV']] = s['obj'].I( V[s['idV']], S[s['idS']],t)
        return I



    def dS(self,V,S):
        """return the differential of the state variables of the ion channels"""
        dS = np.zeros_like(S)
        for s in self.sections:
            dS[s['idS']] = s['obj'].dS(V[s['idV']] + self.V_ref, S[s['idS']])
        return dS





    def indexV(self):
        """return index for potential vector
        id section
        position
        """
        return (np.concatenate( [ [s['num']]*len(s['idV']) for s in self.sections]),
                np.concatenate([s['obj'].x for s in self.sections]))

    def indexS(self):
        """return index for state variables
        id section
        channel
        variable
        position
        """
        #todo
        return (np.concatenate( [ [s['num']]*len(s['idS']) for s in self.sections]),
                np.concatenate([s['obj'].index for s in self.sections]))






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
