# coding: utf-8
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

from copy import deepcopy
import warnings

import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import int8, float32
from quantiphy import Quantity
from scipy.sparse import dia_matrix, csr_matrix
from scipy import interpolate
import param
import networkx as nx

import sinaps.core.species as species

PI = np.pi
SpatialError = ValueError(
    """No spatial resolution for the neuron
                             init_sim(dx) must be called before"""
)


class InitialConcentrationError(ValueError):
    def __init__(self, ion):
        super().__init__(
            """No initial concentration defined for the ion {}
               define it using [section].C0=dict([ion]=[initial_concentration])
                    """.format(
                ion
            )
        )


class DiffusionCoefficientError(ValueError):
    def __init__(self, ion):
        super().__init__(
            """No diffusion coeficient defined for the ion {}
               define it using [section].D=dict([ion]=[initial_concentration])
                    """.format(
                ion
            )
        )


class SectionAttribute(param.Range):
    """Parameter than can be a tuple or a value"""

    step = None

    def _validate(self, val):
        if not hasattr(val, "__len__"):
            super()._validate((val,) * 2)
        else:
            super()._validate(val)


class Neuron(param.Parameterized):
    """Set of sections connected together.

    Attributes
    ----------
    sections : dict(Section), optional
        `sections` is the structure the neuron : It is a dict with section
        as key and  a tuple of integer as value for identifying the nodes
    V_ref : float, optional
        Resting potential of the neuron due to long term dynamic of ions
        channels not modeled


    Examples
    --------
    Create an empty neuron:

    >>> nrn = Neuron()

    Create a neuron with one section binding nodes 0 and 1:

    >>> nrn = Neuron([(0,1)])

    or specifying the section:

    >>> sec = Section()
    >>> nrn = Neuron({sec: (0,1)})

    """

    DEFAULT_CONCENTRATION = {}
    DEFAULT_DIFFUSION_COEF = {}

    V_ref = param.Number(0, doc="Resting potential of the neuron [mV]")
    reactions = param.List(
        [],
        doc="Chemical reactions to consider in the reaction-electrodiffusion simulation",
    )

    def __init__(self, sections=None, **kwargs):

        super().__init__(**kwargs)
        self._species = set()
        self.dx = None
        self._mat = None
        self._x = None
        self._y = None
        self._graph = nx.Graph()
        self.__traversal_source__ = None
        self.__traversal_func__ = nx.dfs_edges
        if sections is not None:
            if issubclass(type(sections), dict):
                self.add_sections_from_dict(sections)
            else:
                self.add_sections_from_list(sections)
            self.__traversal_source__ = list(self.sections.values())[0][0]

    @property
    def graph(self):
        return self._graph

    @property
    def sections(self):
        G = self.graph
        if self.__traversal_func__ is None:
            edges = G.edges
        else:
            edges = self.__traversal_func__(G, self.__traversal_source__)
        return {G.edges[e]["section"]: e for e in edges}

    def add_section(self, sec, i, j):
        """Connect nodes.

        example : N.add_section(s,i,j)
        With i and j:Int and s:Section
        connect nodes i and j with section s
        """
        if not issubclass(type(sec), Section):
            raise ValueError("Must be a section (sinaps.Section)")
        self.graph.add_edge(i, j, section=sec)
        if self.__traversal_source__ is None:
            self.__traversal_source__ = i

    def add_sections_from_dict(self, sections):
        """Add multiple sections from dictionnary {sec:(i,j)}."""
        for sec, (i, j) in sections.items():
            self.add_section(sec, i, j)

    def add_sections_from_list(self, sections):
        """Add multiple sections from a list {(i,j)}."""
        for i, j in sections:
            self.add_section(Section(), i, j)

    def add_species(self, species, C0=None, D=None):
        """Add species to the neuron.

        Set the paramaters C0 (initial concentration) and D (Diffusion
        coeficient) for each section

        Parameters
        ----------
        species : Species
        C0 : number or dict [mMol/L], optional
            Default value for initial concentration if not defined in the
            section. The default is None.
        D : number or dict [𝜇𝑚^2/ms], optional
            Default value for difusion coef if not defined in the section.
            The default is None.

        Raises
        ------
        InitialConcentrationError

        Examples
        -------
        Add species `Ca` :

        >>> nrn.add_species(Species.Ca)

        Add species `Ca` with initial concentration 1 mMol/L and difusion coeficient  1 𝜇𝑚^2/ms:

        >>> nrn = Neuron([(0,1)])
        >>> nrn.add_species(Species.Ca, C0=1, D=1)


        """
        C0 = C0 if C0 is not None else self.DEFAULT_CONCENTRATION
        D = D if D is not None else self.DEFAULT_DIFFUSION_COEF

        if not hasattr(species, "__iter__"):
            if type(C0) in (float, int):
                C0 = {species: C0}
            if type(D) in (float, int):
                D = {species: D}
            species = {species}

        self._species = self._species.union(species)
        for sp in species:
            for sec in self.sections:
                if not (sp in sec.C0):
                    if sp in C0:
                        sec.C0[sp] = C0[sp]
                    else:
                        raise InitialConcentrationError(sp)
                if not (sp in sec.D):
                    if sp in D:
                        sec.D[sp] = D[sp]
                    else:
                        raise DiffusionCoefficientError(sp)

    def add_reaction(self, left, right, k1, k2):
        """Add chemical reaction to the Neuron.

        The reaction is defined with the chemical equation and the reactions
        rates.
        The reaction will be simulated during the electrodiffusion simulation
        Species are automatically added to the neuron if they are not already
        present.

        Parameters
        ----------
        left, right : dict(Species:number)
            chemical equation, with species as key and stoichiometric
            coefficients as value
        k1, k2 : float
            reaction rates, units depends of stoichiometric coeficents, base
            value are ms and mMol/L

        """
        self.reactions.append((left, right, k1, k2))
        self.add_species(list(left.keys()) + list(right.keys()))

    @property
    def species(self):
        """`species` is a set of species for electrodiffusion simulation."""
        return self._species

    @property
    def nb_nodes(self):
        """Number of nodes of the neuron"""
        return max([max(ij) for ij in self.sections.values()]) + 1

    def _init_sim(self, dx, force=False):
        """Prepare the neuron to run a simulation with spatial resolution dx."""
        self.nb_comp = 0  # number of comparment
        self.dx = dx
        for s in self.sections:
            self.nb_comp += s._gen_comp(dx, force)

        idV0 = 0
        idS0 = self.nb_comp
        for s in self.sections:
            idV0, idS0 = s._init_sim(idV0, idS0)

        self.idV = np.array(range(idV0), int)
        self.idS = np.array(range(self.nb_comp, idS0), int)

        return self.idV, self.idS

    def _all_channels(self):
        """Return channels objects suitable for the simulation."""
        cch = {}
        for ch_cls in {type(c) for s in self.sections for c in s.channels}:
            ch = _SimuChannel(ch_cls)
            params = {p: [] for p in ch_cls.param_names}
            idV, idS, surface = [], [], []
            for s in self.sections:
                for c in s.channels:
                    if type(c) is ch_cls:
                        for p in params:
                            if hasattr(c, "position"):  # Point channel
                                value = c.params[p]
                            else:  # Continuous channel
                                value = s._param_array(c.params[p])
                            params[p].append(value)
                        idV.append(c.idV)
                        idS.append(c.idS)
                        if hasattr(c, "position"):  # Point channel
                            surface.append([1])
                        else:  # Continuous channel
                            surface.append(s._surface_array())
            ch.params = {k: np.hstack(v)[:, np.newaxis] for k, v in params.items()}
            ch.idV = np.hstack(idV)
            ch.nb_var = ch_cls.nb_var
            if ch.nb_var:
                ch.idS = [np.hstack(idS)[k, :] for k in range(ch.nb_var)]
            else:
                ch.idS = []
            ch.k = np.concatenate(surface)[:, np.newaxis]
            cch[ch_cls] = ch
        return cch

    def _all_Vsource(self):
        """Return voltage source objects suitable for the simulation."""
        v_source = [c for s in self.sections for c in s.Vsource]
        row = np.array([v.idV for v in v_source], int)
        col = np.ones((1, len(v_source)), int)
        data = np.ones_like(row)
        col = np.zeros_like(row)
        source_mat = csr_matrix((data, (row, col)), (self.nb_comp, 1))
        return v_source, source_mat

    def _capacitance_array(self):
        """Return the menbrane capacitance for each nodes.

        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise SpatialError

        Cm = np.zeros(self.nb_comp)
        for s in self.sections:
            Cm[s.idV] = s._c_m_array()

        return Cm

    def _volume_array(self):
        """Return the volume of each nodes.

        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise SpatialError

        V = np.zeros(self.nb_comp)
        for s in self.sections:
            V[s.idV] = s._volume_array()

        return V

    def radius_array(self):
        """Return the radius of each nodes.

        init_sim(dx) must have been previously called
        """
        a = np.zeros(self.nb_comp)
        for s in self.sections:
            a[s.idV] = s._param_array(s.a)
        return a

    def _conductance_mat(self):
        """Return conductance matrix G of the neuron.

        G is a sparse matrix
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise SpatialError

        n = self.nb_comp
        m = self.nb_nodes

        # Initilialisation conductance matrix size n * n+m
        G = np.zeros([n, n + m])

        for s, (i, j) in self.sections.items():
            # longitudinal conductance in a section
            g_l = 1 / s._r_l_array()
            G[s.idV[:-1], s.idV[1:]] += g_l
            G[s.idV[1:], s.idV[:-1]] += g_l
            G[s.idV[:-1], s.idV[:-1]] += -g_l
            G[s.idV[1:], s.idV[1:]] += -g_l
            #  conductance between connecting nodes and first/last compartment
            g_end = 1 / s._r_l_end()
            ends = [0, -1]
            G[s.idV[ends], [i + n, j + n]] += g_end
            G[s.idV[ends], s.idV[ends]] += -g_end

        return G

    def _connection_mat(self):
        """Return connection matrix k of the neuron.

        Giving the relation between the voltage of the connecting nodes and the
        voltage of the compartiment nodes
        init_sim(dx) must have been previously called

        V0 = sum (gk.Vk) / sum(gk)

        """
        if self.dx is None:
            raise SpatialError

        # Initilialisation connection matrix size m * n
        k = np.zeros([self.nb_nodes, self.nb_comp])

        for s, (i, j) in self.sections.items():
            g_end = 1 / s._r_l_end()
            # conductance of leak channels
            k[[i, j], s.idV[[0, -1]]] = g_end
        k = k / k.sum(axis=1, keepdims=True)
        return k

    def _fill_V0_array(self, V0):
        """Fill the initial potential for each nodes.

        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise SpatialError

        for s in self.sections:
            s._fill_V0_array(V0)
        # Conversion of reference potential V=0 for the resting potential
        V0[self.idV] = V0[self.idV] - self.V_ref

    def _fill_S0_array(self, S0):
        """Fill initial state variable for each nodes.

        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise SpatialError

        for s in self.sections:
            s._fill_S0_array(S0)

    def _fill_C0_array(self, C0, ions):
        """Return the initial concentration for each nodes.

        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise SpatialError

        for s in self.sections:
            s._fill_C0_array(C0, ions)

    # Diffusion functions

    def _difus_array(self, ion):
        """Return the diffusion coefficient D*a/dx between each node for ion.

        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise SpatialError
        cc = np.concatenate
        return cc([cc([[0], s._difus_array(ion)]) for s in self.sections])

    def _difus_end(self, ion):
        """Return the diffusion coefficient D*a/dx.

        between the start/end of the section and the first/last node for ion
        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        if self.dx is None:
            raise SpatialError

        return (
            np.array([s._difus_end(ion) for s in self.sections])
            .swapaxes(0, 1)
            .flatten()
        )
        # return the firs node of all section first
        # then the last node of all sections

    def _difus_mat(self, T, ion, Vt):
        """Return the electrodiffusion matrix.

        init_sim(dx) must have been previously called

        Parameters
        ----------
        T : number
            Temperature [K].
        ion : _Species
            DESCRIPTION.
        Vt : [float32:]
            Potential *V* [mV] for each compartment.

        Returns
        -------
        [float32:]
            Electrodiffusion matrix.

        """
        cc = np.concatenate
        if self.dx is None:
            raise SpatialError
        n = self.nb_comp
        m = self.nb_nodes

        Vtt = Vt[1:] - Vt[:-1]

        # e=1.6021766208e-19 A.s electron charge
        # e=1.6021766208e-4 pA.ms
        # kb=1.38064852e-23 J.K-1 = Ω.A^2.s Boltzman constant
        # kb=1.38064852e-05 GΩ.pA^2.ms
        # e/kb=11.6045 1/(pA.GΩ)=mV^-1
        # k=ez/kbT
        k = 11.6045 * species.CHARGE[ion] / T

        # flux inside a section
        d = np.resize(self._difus_array(ion), n + m)  # [μm^3/ms]
        j = dia_matrix(
            (
                np.array(
                    [
                        cc([[0], d[1:] * (1 + k / 2 * Vtt)]),
                        cc([d[1:] * (-1 + k / 2 * Vtt), [0]]),
                    ]
                ),
                [0, -1],
            ),
            shape=(n + 1, n + m),
        ).tolil()
        j[-1] = 0  # No flux for last compartiment
        jtt = (j[1:] - j[:-1]).tolil()  # flux difference [μm^3/ms]

        # flux for connecting nodes
        d_end = self._difus_end(ion)
        idA = cc(
            [[s.idV[0] for s in self.sections], [s.idV[-1] for s in self.sections]]
        )
        idB = cc(
            [
                [i + n for s, (i, j) in self.sections.items()],
                [j + n for s, (i, j) in self.sections.items()],
            ]
        )
        Vttk = k / 2 * (Vt[idA] - Vt[idB])
        jtt[idA, idB] = d_end * (1 - Vttk)
        jtt[idA, idA] += d_end * (-1 - Vttk)
        return jtt  # [μm^3/ms]

    def indexV(self, species=None):
        """return index for potential vector
        id section
        position
        """
        if species is None:
            return (
                np.concatenate([[s.name] * len(s.idV) for s in self.sections]),
                np.concatenate([s.x for s in self.sections]),
            )
        else:
            return (
                np.concatenate([[sp] * self.nb_comp for sp in species]),
                np.concatenate(
                    [[s.name] * len(s.idV) for s in self.sections] * len(species)
                ),
                np.concatenate([s.x for s in self.sections] * len(species)),
            )

    def indexV_flat(self):
        """return position (flattened) for potential vector"""
        x = 0
        ind = []
        for s in self.sections:
            ind = np.append(ind, s.x + x)
            x += s.L
        return ind

    def indexS(self):
        """return index for state variables
        id section
        channel
        variable
        position
        """
        # todo
        return (
            np.concatenate([[s.name] * len(s.idS) for s in self.sections]),
            np.concatenate([s.index for s in self.sections]),
        )


class Section(param.Parameterized):
    """Part of neuron with uniform physical values

    Resistance and capacitance given per membrane-area unit
    """

    _next_id = 0

    L = param.Number(
        100, bounds=(0, None), inclusive_bounds=(False, True), doc="length [μm]"
    )
    a = SectionAttribute(
        1, bounds=(0, None), inclusive_bounds=(False, True), doc="radius [μm]"
    )
    C_m = SectionAttribute(
        1,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="menbrane capacitance [μF/cm²]",
    )
    R_l = SectionAttribute(
        150,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="longitudinal resistance [Ohm.cm]",
    )
    V0 = param.Number(0, doc="initial potential [mV]")
    C0 = param.Dict({}, doc="initial concentration for species [mM/L]")
    D = param.Dict({}, doc="diffusion coeficient for species [um2/ms]")
    dx = param.Number(
        None,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="Spatial resolution for the simulation. If None the global spatial resolution of the Simulation object is used",
    )

    def __init__(self, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "Section{:04d}".format(Section._next_id)
        super().__init__(**kwargs)
        Section._next_id += 1
        self.channels_c = []
        self.channels_p = []
        self.Vsource = []

    @property
    def _C_m(self):
        """Menbrane capacitance [pF/um²]"""
        return self.C_m / 100

    @property
    def _R_l(self):
        """longitudinal resistance [GOhm.um]"""
        return self.R_l / 1e5

    @property
    def r_l(self):
        """Longitudinal resistance per unit length [GOhm/μm]"""
        return self._R_l / (PI * self.a ** 2)

    @property
    def c_m(self):
        """Membrane capacitance per unit length [pF/μm]"""
        return self._C_m * 2 * PI * self.a

    def __str__(self):
        return "Section {}".format(self.name)

    def _repr_markdown_(self):
        return """Section **{}**

* L: {}
* a: {}
* C_m: {} (c_m={})
* R_l: {} (r_l={})
* channels: {}
* point_channels: {}""".format(
            self.name,
            Quantity(self.L * 1e-6, "m"),
            Quantity(self.a * 1e-6, "m"),
            Quantity(self.C_m * 1e-6, "F/cm²"),
            Quantity(self.c_m * 1e-12, "F/μm"),
            Quantity(self.R_l, "Ω.cm"),
            Quantity(self.r_l * 1e-12, "Ω/μm"),
            "\n  + ".join([str(c) for c in self.channels_c]),
            "\n  + ".join(["{}:{}".format(c.position, c) for c in self.channels_p]),
        )

    def add_channel(self, C, x=None):
        """Add a channel to the section

        Parameters
        ----------
        C: Channel
        x: Float, optional
            relative position along the section (between 0 and 1),
            if None (default) the channels is treated as a continuous density channel
            giving a surfacic current in nA/μm2

        """
        if not issubclass(type(C), Channel):
            raise ValueError("Must be a channel (sinaps.Channel)")

        if type(C) in [type(c) for c in self.channels]:
            warnings.warn(
                ("The section already had a channel of type {} ".format(type(C)))
            )

        C = deepcopy(C)
        if x is None:
            self.channels_c.append(C)
        else:
            C.position = x
            self.channels_p.append(C)

    def clear_channels(self):
        """Clear all channels"""
        self.channels_c = []
        self.channels_p = []

    def add_voltage_source(self, C, x=0):
        """Add a voltage source to the section

        Parameters
        ----------
        V: VoltageSource
        x : Flaot, optional
            relative position along the section (between 0 and 1),

        """
        if not issubclass(type(C), VoltageSource):
            raise ValueError("Must be a channel (sinaps.VoltageSource)")
        C = deepcopy(C)
        C.position = x
        self.Vsource.append(C)

    @property
    def channels(self):
        return self.channels_c + self.channels_p

    # Initilialisation functions
    def _gen_comp(self, dx, force=False):
        """section.dx will be use if existing expect if force is True"""
        if self.dx is None or force:
            self._dx = dx
        else:
            self._dx = self.dx
        try:
            self.nb_comp = max(int(np.ceil(self.L / self._dx)), 2)
        except ZeroDivisionError:
            raise ValueError("dx must be defined for section {}".format(self.name))
        return self.nb_comp

    def _init_sim(self, idV0, idS00):
        """Prepare the section to run a simulation with spatial resolution dx
        gen_comp must have been run before
        idV0 : first indices for Voltage
        idS00 : first indice for state variables
        """
        n = self.nb_comp
        self._dx = self.L / n
        # center of the compartiment, size n
        self.x = np.linspace(self._dx / 2, self.L - self._dx / 2, n)
        # border of the compartiment, size n+1
        self.xb = np.linspace(0, self.L, n + 1)
        self.idV = idV0 + np.array(range(n), int)
        idS0 = idS00
        for c in self.Vsource:
            c.idV = idV0 + int(min(c.position * n, n - 1))
        for c in self.channels_c:
            c.idS = [k * n + idS0 + np.array(range(n), int) for k in range(c.nb_var)]
            c.idV = self.idV
            idS0 += n * c.nb_var  # * state variables
        for c in self.channels_p:
            # index of the compartment related to the point process
            c.idV = idV0 + int(min(c.position * n, n - 1))
            c.idS = [k + idS0 + np.array([0], int) for k in range(c.nb_var)]
            idS0 += c.nb_var  # * state variables
        self.nb_var = idS0 - idS00
        return idV0 + n, idS0

    def _continuous(self, param):
        if callable(param):
            return np.vectorize(param)
        elif issubclass(type(param), tuple):
            return interpolate.interp1d([0, self.L], param)
        else:
            return lambda x: param * np.ones_like(x)

    def _param_array(self, param):
        return self._continuous(param)(self.x)

    def _param_array_diff(self, param):
        return self._continuous(param)((self.x[:-1] + self.x[1:]) / 2)

    def _param_array_end(self, param):
        return self._continuous(param)([self.x[0] / 2, (self.L + self.x[-1]) / 2])

    def _fill_V0_array(self, V0):
        """fill the initial potential for each nodes
        init_sim(dx) must have been previously called
        """
        V0[self.idV] = self._param_array(self.V0)  # [mV]

    def _fill_S0_array(self, S0):
        """fill the initial state variables values for each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        for c in self.channels:
            if c.nb_var:
                S0[np.hstack(c.idS)] = c.__S0(len(self.x))

    def _fill_C0_array(self, C0, ions):
        """Return the initial concentration for each nodes
        init_sim(dx) must have been previously called
        """
        for k, ion in enumerate(ions):
            if ion in self.C0:
                C0[self.idV, k] = self._param_array(self.C0[ion])  # [mM/L]
            else:
                raise InitialConcentrationError(ion)

    def _c_m_array(self):
        """Return the membrane capacitance for each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        return self._surface_array() * self._param_array(self._C_m)  # [nF]

    def _volume_array(self):
        """Return the volume of each compartment
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        return np.diff(self.xb) * self._param_array(self.a) ** 2 * PI  # [um3]

    def _surface_array(self):
        """Return the menbrane surface of each compartment
        init_sim(dx) must have been previously called
        """
        # np.diff(self.xb) = size of compartiment
        return np.diff(self.xb) * self._param_array(self.a) * 2 * PI  # [um2]

    def _r_l_array(self):
        """Return the longitudinal resistance between each nodes
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        return (
            np.diff(self.x)
            * self._param_array_diff(self._R_l)
            / (self._param_array_diff(self.a) ** 2 * PI)
        )  # [MΩ]

    def _r_l_end(self):
        """Return the longitudinal resistance between the start/end of the section
        and the first/last node
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        return (
            np.array([self.x[0], (self.L - self.x[-1])])
            * self._param_array_end(self._R_l)
            / (self._param_array_end(self.a) ** 2 * PI)
        )  # [MΩ]

    # Diffusion functions
    def _difus_array(self, ion):
        """Return the diffusion coefficient D*a/dx between each node for ion
        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        return (
            self._param_array_diff(self.D[ion])  # [μm^2/ms]
            * PI
            * self._param_array_diff(self.a) ** 2  # [μm]
        ) / np.diff(
            self.x
        )  # [μm^3/ms]

    def _difus_end(self, ion):
        """Return the diffusion coefficient D*a/dx between the start/end of the section
        and the first/last node for ion
        unit μm^3/ms
        init_sim(dx) must have been previously called
        """
        # np.diff(self.x) = distance between centers of compartiment
        return (
            self._param_array_end(self.D[ion])  # [μm^2/ms]
            * PI
            * self._param_array_end(self.a) ** 2  # [μm]
        ) / np.array(
            [self.x[0], (self.L - self.x[-1])]
        )  # [μm^3/ms]

    def __lt__(self, other):
        return self.name < other.name


class Channel:
    """Generic class describing a channel or a group of channel.

    To implement a channel `C`, it is necessary to implement:

    + a function `C.I(V,*st_vars)` returning the net  current towards inside
      ([pA] for point channel and [pA/μm2] for density channels ) of the mechanism with
      `V` the voltage (:array) and `*st_vars` the state variable of the mechanism (:array)
    + If there is state variables a function `C.dS(V,*st_vars)` returning a
      tuple with the differential of each state variable
      and a function `S0` returning a tuple with the initial value for each state
      variable

    """

    nb_var = 0
    param_names = ()
    params = {}

    def I(self, V, *S, t):
        return self._I(V, *S, t=t, **self.params)

    def J(self, V, *S, t):
        return self._I(V, *S, t=t, **self.params)

    def dS(self, V, *S, t=0):
        return self._I(V, *S, t=t, **self.params)

    @staticmethod
    def _I(V, t):
        return 0 * V

    @staticmethod
    def _dS(V, *S, t, **unused):
        return (0,) * len(S)

    def S0(self):
        return (0,) * self.nb_var

    def _Section__S0(self, n):
        if self.nb_var > 1:
            return np.concatenate([[S] * n for S in self.S0()])
        else:
            return [self.S0()] * n

    def density_str(self):
        if hasattr(self, "position"):
            return ""
        else:
            return "/μm²"


class _SimuChannel:
    """Class for vectorization of channels used for efficiency"""

    def __init__(self, ch_cls):
        self.I = njit(ch_cls._I)
        self.dS = njit(ch_cls._dS)
        if hasattr(ch_cls, "_J"):
            self.J = njit(ch_cls._J)
        self.type = ch_cls

    def __repr__(self):
        return "<SimuChannel : {} ( {} comp)>".format(self.type.__name__, len(self.idV))

    def fill_I_dS(self, y, V_S, t):
        """fill the array y with :
        - The transmembrane curent of the channels self
        - the variation of the states variable of the channels self
        V_S : current values for potential and state variables
        t : current time (ms)
        """
        V = V_S[self.idV, :]
        S = [V_S[self.idS[k], :] for k in range(self.nb_var)]
        np.add.at(y, np.s_[self.idV, :], self.I(V, *S, t, **self.params) * self.k)
        if self.nb_var:
            dS = self.dS(V, *S, t, **self.params)
            if self.nb_var > 1:
                for k in range(self.nb_var):
                    y[self.idS[k], :] = dS[k]
            else:
                y[self.idS[0], :] = dS

    def fill_J(self, y, ions, V_S, t):
        """fill the array y with :
        - The flux of ion of the channels self
        V_S : current values for potential and state variables
        t : current time (ms)"""
        # continuous channels
        if hasattr(self, "J"):
            V = V_S[self.idV]
            S = [V_S[self.idS[k]] for k in range(self.nb_var)]
            for k, ion in enumerate(ions):
                np.add.at(
                    y,
                    np.s_[self.idV, k],
                    (self.J(ion, V, *S, t, **self.params) * self.k).squeeze(),
                )


class VoltageSource:
    """Represents a Voltage Source V = f(t)"""

    def __init__(self, f):
        self.V = njit(f)


@jitclass({"charge": int8, "D": float32})
class _Species:
    """Represents a species."""

    def __init__(self, charge, D):
        """
        charge : int electric charge of the ion (ex 2 for Ca2+ and -1 for Cl-)
        D : diffusion coef of the ion um2/ms
        """
        self.charge = charge
        self.D = D


# %% Tests

if __name__ == "__main__":
    import doctest

    doctest.testmod(
        extraglobs={
            "nrn": Neuron(),
            "Species": species.Species,
        }
    )
