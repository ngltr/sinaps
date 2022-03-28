# coding: utf-8
from functools import lru_cache
from functools import reduce

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, dia_matrix
from scipy import interpolate
from numba import jit
from tqdm import tqdm

from sinaps.core.species import CHARGE


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
        self.V_S0 = np.zeros(max(np.concatenate((self.N.idV, self.N.idS))) + 1)
        self.N._fill_V0_array(self.V_S0)
        self.N._fill_S0_array(self.V_S0)
        self.channels = self.N._all_channels()
        self.ions = 0
        self.C = dict()
        self.sol_diff = dict()
        self.progressbar = progressbar

    def run_c(
        self,
        t_span,
        species=None,
        temperature=310,
        method="BDF",
        atol=1.49012e-8,
        **kwargs
    ):
        """Run the voltage related simulation

        The results of the simulation are stored in attribute `V`

        Parameters
        ----------
        t_span : 2-tuple of number
            Timeframe for the simulation (ms)

        Other Parameters
        ----------------
        **kwargs :
            args to pass to the ode solver. see the `scipy solve_ivp <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_ doc

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
        y0 = np.hstack([self.V_S0, np.reshape(C0, -1, "F")])
        idC = np.array(range(len(self.V_S0), len(y0)))
        charges = (
            np.array([CHARGE[s] for s in species]) * 96.48533132838746
        )  # pA/aMol = C/mol/ms E-3

        sol = solve_ivp(
            lambda t, y: self._ode_function(
                y,
                t,
                self.idV,
                self.idS,
                idC,
                self.Cm1,
                self.G,
                self.channels.values(),
                species,
                charges,
                reactions,
                temperature,
                self.N.nb_comp,
                tq,
                t_span,
            ),
            t_span,
            y0,
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

        tq = tqdm(total=self.t_span[1] - self.t_span[0], unit="ms")

        sp, sec, pos = self.N.indexV(species)
        df = pd.DataFrame(sol.y[idC, :].T, sol.t, [sp, sec, pos])
        df.columns.names = ["Species", "Section", "Position (μm)"]
        df.index.name = "Time"
        self.C = df
        self.sol_diff = sol

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

    def _ode_function(
        self,
        y,
        t,
        idV,
        idS,
        idC,
        Cm1,
        G,
        channels,
        ions,
        charges,
        reactions,
        T,
        nb_comp,
        tq=None,
        t_span=None,
    ):
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
        V = y[idV]
        y = y[:, np.newaxis]

        dV_S_C = np.zeros_like(y)
        for c in channels:
            c.fill_I_dS(
                dV_S_C, y, t
            )  # current of active ion channels from outisde to inside

        C = np.reshape(y[idC, :], (nb_comp, len(ions)), "F")
        dC = np.zeros_like(C)

        if len(ions) > 1:
            dC += np.vstack(
                [
                    (self.N._difus_mat(T, ion, self.k_c @ V) @ self.k_c) @ C[:, k]
                    for k, ion in enumerate(ions)
                ]
            ).T
        else:
            dC += self._difus_mat(T, ions[0], t) @ C

        J = np.zeros((self.N.nb_comp, len(ions)))
        for c in self.channels.values():
            c.fill_J(J, ions, y, t)
        dC += J  # [aM/ms]
        dV_S_C[idV, :] = (dC @ charges)[:, np.newaxis]  # pA
        dC *= self.Vol1  # [aM/μm^3/ms]
        # _fill_dC_reaction(dC,C,reactions)
        dV_S_C[idC, :] = np.reshape(dC, -1, "F")[:, np.newaxis]
        dV_S_C[idV, :] *= Cm1  # dV/dt for  compartiment #mV

        # Progressbar
        if not (tq is None):
            n = round(t - t_span[0], 3)
            if n > tq.n:
                tq.update(n - tq.n)

        return dV_S_C.squeeze()

    def resample(self, freq):
        self.V = self.V.resample(freq).mean()


# %%
if __name__ == "__main__":
    import sinaps as sn

    # %%
    nrn = sn.Neuron([(0, 1), (1, 2), (1, 3), (3, 4)])
    nrn[
        0
    ].clear_channels()  # Reset all channels, useful if you re-run this cell (does nothing the first time)
    nrn[0].add_channel(sn.channels.PulseCurrent(500, 2, 3), 0)
    nrn[0].add_channel(sn.channels.Hodgkin_Huxley())
    nrn[0].add_channel(sn.channels.Hodgkin_Huxley_Ca(gCa=14.5, V_Ca=115))
    sim = Simulation(nrn, dx=10)
    # shortcuts
    Ca = sn.Species.Ca
    BF = sn.Species.Buffer
    BFB = sn.Species.Buffer_Ca
    nrn.add_reaction({Ca: 1, BF: 1}, {BFB: 1}, k1=0.2, k2=0.1)
    # Calcium extrusion
    nrn.add_reaction({Ca: 1}, {}, k1=0.05, k2=0)
    # Calcium initial concentration
    for s in nrn.sections:
        s.C0[Ca] = 10 ** (-4)
        s.C0[BF] = 2 * 10 ** (-3)

    sim.run((0, 100))
