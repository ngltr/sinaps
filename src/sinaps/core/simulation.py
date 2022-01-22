# coding: utf-8
from functools import lru_cache
from functools import reduce

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, bmat
from scipy import interpolate
from scipy.misc import derivative
from numba import jit
from tqdm import tqdm


class Simulation:
    """
    Object used for running simulations and accessing results

    The class simulation is linked to a specific neuron and is used to run voltage propagation simulation and electrodiffusion simulations with some custom spatial and time resolution.
    The results of the simulation are stored as attributes

    Parameters
    ----------

    neuron : sinaps.Neuron
        Neuron used for the simulation

    dx : float
        Spatial resolution (um) to used if no specific dx is set for the section

    force_dx : bool
        If True, `dx` will be used even if a custom dx was set for a section


    Examples
    --------

    Create the simulation with neuron `N` and spatial resolution of `20 um` :

    >>> sim=sn.Simulation(N,dx=20)

    """

    def __init__(self, neuron, dx=None, force_dx=False, progressbar=tqdm):
        self.N = neuron
        self.idV, self.idS = self.N._init_sim(dx, force_dx)
        self.Cm1 = 1 / self.N._capacitance_array()[:, np.newaxis]
        self.Vol1 = 1 / self.N._volume_array()[:, np.newaxis]
        G = csr_matrix(self.N._conductance_mat())  # sparse matrix format for
        # efficiency
        self.k_c = csr_matrix(
            np.concatenate([np.identity(self.N.nb_comp), self.N._connection_mat()])
        )
        self.G = G @ self.k_c
        self.v_source, source_mat = self.N._all_Vsource()
        # if len(self.v_source):
        #    self.G = bmat([[self.G, source_mat], [source_mat.T, None]]).tocsr()
        self.V_S0 = np.zeros(max(np.concatenate((self.N.idV, self.N.idS))) + 1)
        self.N._fill_V0_array(self.V_S0)
        self.N._fill_S0_array(self.V_S0)
        self.channels = self.N._all_channels()
        self.ions = 0
        self.C = dict()
        self.sol_diff = dict()
        self.progressbar = progressbar

    def run(self, t_span, method="BDF", atol=1.49012e-8, **kwargs):
        """Run the voltage related simulation

        The results of the simulation are stored in attribute `V`

        Parameters
        ----------
        t_span : 2-tuple of number
            Timeframe for the simulation (ms)

        Other Parameters
        ----------------
        **kwargs :
            args to pass to the ode solver. see the `scipy solve_ivp
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_ doc

        Examples
        --------
        Run the simulation between 0 and 10 ms :

        >>> sim.run((0,10))

        View the results (potential vector for each time and position):

        >>> sim.V

        """
        if self.progressbar is not None:
            tq = self.progressbar(total=t_span[1] - t_span[0], unit="ms")
        else:
            tq = None

        for v in self.v_source:
            self.V_S0[v.idV] = v.V(t_span[0])

        sol = solve_ivp(
            lambda t, y: Simulation._ode_function(
                y,
                t,
                self.idV,
                self.idS,
                self.Cm1,
                self.G,
                self.v_source,
                self.channels.values(),
                tq,
                t_span,
            ),
            t_span,
            self.V_S0,
            method=method,
            atol=atol,
            **kwargs
        )
        if tq is not None:
            tq.close()

        sec, pos = self.N.indexV()
        df = pd.DataFrame(sol.y[: self.N.nb_comp, :].T, sol.t, [sec, pos])
        df.columns.names = ["Section", "Position (μm)"]
        df.index.name = "Time"
        self.V = df + self.N.V_ref
        self.sol = sol
        # sec,pos = self.N.indexS()
        df = pd.DataFrame(sol.y[self.idS, :].T, sol.t)  # ,[sec, pos * 1E6])
        # df.columns.names = ['Section','Channel','Variable,''Position (μm)']
        df.index.name = "Time (ms)"
        self.S = df
        self.t_span = t_span
        self.V_St = interpolate.interp1d(
            self.sol.t, self.sol.y, fill_value="extrapolate"
        )

    def current(self, ch_cls):
        """Return current of channels of a simulations

        Parameters
        ----------
        ch_cls: type
            Class of channel

        Returns
        --------
        a panda.DataFrame with current for all time and compartment

        Examples
        --------
        Get the Hodgkin_Huxley current :

        >>> sim.current(sn.channels.Hodgkin_Huxley)

        """
        ch = self.channels[ch_cls]
        t = self.sol.t
        V_S = self.V_St(t)
        I = np.zeros_like(V_S)
        ch.fill_I_dS(I, V_S, t)
        sec, pos = self.N.indexV()
        df = pd.DataFrame(I[: self.N.nb_comp].T, t, [sec, pos])
        df.columns.names = ["Section", "Position (μm)"]
        df.index.name = "Time"
        return df

    @staticmethod
    def _ode_function(y, t, idV, idS, Cm1, G, v_source, channels, tq=None, t_span=None):
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

        """
        vectorize = y.ndim > 1
        if not vectorize:
            y = y[:, np.newaxis]

        V = y[idV, :]
        for v in v_source:
            V[v.idV, :] = v.V(t)
        dV_S = np.zeros_like(y)
        for c in channels:
            c.fill_I_dS(
                dV_S, y, t
            )  # current of active ion channels from outisde to inside
        if vectorize:
            dV_S[idV, :] += np.hstack(
                [G @ V[:, k] for k in range(y.shape[1])]
            )  # current inter compartment
        else:
            dV_S[idV, :] += G @ V
        dV_S[idV, :] *= Cm1  # dV/dt for  compartiment
        for v in v_source:
            dV_S[v.idV, :] = derivative(v.V, t, 1e-3)
        # Progressbar
        if not (tq is None):
            n = round(t - t_span[0], 3)
            if n > tq.n:
                tq.update(n - tq.n)

        return dV_S.squeeze()

    def run_diff(
        self, species=None, temperature=310, method="BDF", atol=1.49012e-8, **kwargs
    ):
        """Run the electro-diffusion simulation based on Nersnt-Planck model.

        The simulation for the voltage must have been run before.

        The results of the simulation are stored in attribute `C`

        Parameters
        ----------
        species : [species]
            Species to consider in simulation
        temperature : float
            Temperature in Nerst Planck equation

        Other Parameters
        ----------------
        **kwargs :
            args to pass to the ode solver. see the `scipy solve_ivp
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_ doc

        """
        tq = tqdm(total=self.t_span[1] - self.t_span[0], unit="ms")

        Simulation._flux.cache_clear()
        Simulation._difus_mat.cache_clear()

        if species is None:
            species = tuple(
                self.N.species
            )  # conversion in tuple the ensure the order (N.species i a set)

        reactions = []
        for reac in self.N.reactions:
            # Conversion species to int
            members = [
                {species.index(sp): n for sp, n in member.items()}
                for member in reac[0:2]
            ]
            if reac[2]:
                reactions.append((*members, reac[2]))
            members.reverse()
            if reac[3]:
                reactions.append((*members, reac[3]))

        C0 = np.zeros((self.N.nb_comp, len(species)))
        self.N._fill_C0_array(C0, species)

        sol = solve_ivp(
            lambda t, y: self._ode_diff_function(
                y, t, species, reactions, temperature, tq, self.t_span
            ),
            self.t_span,
            np.reshape(C0, -1, "F"),
            method=method,
            atol=atol,
            # jac = lambda t, y:self.jac_diff(y,t,ions,temperature),
            **kwargs
        )
        tq.close()

        sp, sec, pos = self.N.indexV(species)
        df = pd.DataFrame(sol.y.T, sol.t, [sp, sec, pos])
        df.columns.names = ["Species", "Section", "Position (μm)"]
        df.index.name = "Time"
        self.C = df
        self.sol_diff = sol

    def _ode_diff_function(self, y, t, ions, reactions, T, tq=None, t_span=None):
        """this function express the ode problem :
        dy/dt = f(y)

        y is a vector containing all state variables of the problem

        C=y : concentration of each node
        [ aM/μm^3 = m mol/L ] aM : atto(10E-18) mol correction: [ a mol/μm^3 = m mol/L = mM]
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
        C = np.reshape(y, (self.N.nb_comp, len(ions)), "F")
        dC = np.zeros_like(C)

        if len(ions) > 1:
            dC += np.vstack(
                [self._difus_mat(T, ion, t) @ C[:, k] for k, ion in enumerate(ions)]
            ).T
        else:
            dC += self._difus_mat(T, ions[0], t) @ C
        dC += self._flux(ions, t)  # [aM/ms]
        dC *= self.Vol1  # [aM/μm^3/ms]
        _fill_dC_reaction(dC, C, reactions)

        # Progressbar
        if not (tq is None):
            n = round(t - t_span[0], 3)
            if n > tq.n:
                tq.update(n - tq.n)

        return np.reshape(dC, -1, "F")

    def _jac_diff(self, C, t, ions, T):
        D = self._difus_mat(T, ions, t)  # [μm^3/ms]
        return (D @ self.k_c).multiply(self.Vol1)  # [aM/μm^3/ms]

    def resample(self, freq):

        self.V = self.V.resample(freq).mean()

    # Caching functions
    @lru_cache(128)
    def _difus_mat(self, T, ion, t):
        """Return the electrodiffusion matrix for :
        - the ion *ion* type sinaps.Ion
        - potential *V* [mV] for each compartment
        - Temperature *T* [K]
        - *t* : index in array V, usually the time

        (we call with V and t as different argument to be cable to
        #use the caching with lru as this method will be used a lot)
        """
        # We
        return self.N._difus_mat(T, ion, self.k_c @ self.V_St(t)[self.idV]) @ self.k_c

    @lru_cache(128)
    def _flux(self, ions, t):
        """return the transmembrane flux of ion ion (aM/ms attoMol)
                 towards inside

        (we call with V,S and t as different argument to be cable to
        #use the caching with lru as this method will be used a lot)
        """
        J = np.zeros((self.N.nb_comp, len(ions)))
        for c in self.channels.values():
            c.fill_J(J, ions, self.V_St(t)[:, np.newaxis], t)
        return J


def _fill_dC_reaction(dC, C, reactions):
    """Return variation of concentration due to reactions aM/ms"""
    for reaction in reactions:
        dC_reac = (
            reduce(np.multiply, [C[:, sp] ** n for sp, n in reaction[0].items()])
            * reaction[2]
        )
        for sp, n in reaction[0].items():
            # sp :species, n: stoechiometric coefficient
            dC[:, sp] += -n * dC_reac
        for sp, n in reaction[1].items():
            dC[:, sp] += n * dC_reac
