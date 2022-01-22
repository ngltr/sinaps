# coding: utf-8
import numpy as np
from quantiphy import Quantity

from sinaps.core.model import Channel
from sinaps.core.species import Species


class ConstantCurrent(Channel):
    """Channel with a constant current.

    Parameters
    ----------
    current : float
        The current in [pA] for point channel or pA/um2 for density channel

    """

    param_names = ("current",)

    def __init__(self, current):
        self.params = {"current": current}

    @staticmethod
    def _I(V, t, current):
        return current

    def __repr__(self):
        return "ConstantCurrent(I={})".format(
            Quantity(self.params["current"] * 1e-12, "A{}".format(self.density_str()))
        )


class PulseCurrent(Channel):
    """Channel with a constant current between a time period.

    Parameters
    ----------
    current : float
        The current in [pA] for point channel or pA/um2 for density channel
    t0 : float
        [ms] start time of the current pulse
    tf : float
        [ms] end time of the current pulse

    """

    param_names = ("current", "t0", "tf")

    def __init__(self, current, t0, tf):

        self.params = {
            "current": current,
            "t0": t0,
            "tf": tf,
        }

    @staticmethod
    def _I(V, t, current, t0, tf):
        return ((t <= tf) & (t >= t0)) * current


class HeavysideCurrent(PulseCurrent):
    def __init__(self, current, t0, tf):
        super().__init__(current, t0, tf)
        raise FutureWarning("HeavysideCurrent is deprecated, use PulseCurrent instead")


class LeakChannel(Channel):
    """Leak channel  I = (V-Veq) * G_m.

    Parameters
    ----------
    Veq : float
        [mV] Equilibrium potential for the leak channel
    G_m : float
        [mS/cm2] Resistance of the menbrane (leak channel)

    """

    param_names = ("Veq", "R_m")

    def __init__(self, G_m=0.3, Veq=0):
        """ """
        self.params = {"Veq": Veq, "R_m": 1 / G_m * 100}
        # conversion to GΩ.μm2: 1/(1 mS/cm2) = 100 GΩ.μm2

    @staticmethod
    def _I(V, t, Veq, R_m):
        """
        Return the net surfacic current [pA/um2] of the mechanism towards inside
        """
        return (Veq - V) / R_m

    def __repr__(self):
        return "LeakChannel(Veq={}, G_m={})".format(
            Quantity(self.params["Veq"] * 1e-3, "V"),
            Quantity(1 / self.params["R_m"] * 1e-1, "S/cm²"),
        )


class Hodgkin_Huxley(Channel):
    """Hodgkin Huxley channels

    Parameters
    ----------
    gNa : float
        [mS/cm2] Conductance of sodium channel
    V_Na : float
        [mV] Equilibrium potential of sodium channel
    gK :  float
        [mS/cm2] Conductance of potasium channel
    V_K : float
        [mV] Equilibrium potential of potasium channel
    gL :  float
        [mS/cm2] conductance of leak channel
    V_L : float
        [mV] Equilibrium potential of leak channel

    """

    nb_var = 3
    param_names = ("gNa", "V_Na", "gK", "V_K", "gL", "V_L")

    def __init__(self, gNa=120, V_Na=115, gK=36, V_K=-12, gL=0.3, V_L=10.6):
        """Channel Hodgkin Huxley type
        gNa : conductance of sodium channel [mS/cm2]
        V_Na : Equilibrium potential of sodium channel [mV]
        gK :  conductance of potasium channel [mS/cm2]
        V_K : Equilibrium potential of potasium channel [mV]
        gL :  conductance of leak channel [mS/cm2]
        V_L : Equilibrium potential of leak channel [mV]
        """
        self.params = {
            "gNa": gNa / 100,  # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
            "V_Na": V_Na,
            "gK": gK / 100,  # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
            "V_K": V_K,
            "gL": gL / 100,  # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
            "V_L": V_L,
        }

    @staticmethod
    def _I(V, n, m, h, t, gNa, V_Na, gK, V_K, gL, V_L):
        I_Na = gNa * m ** 3 * h * (V - V_Na)
        I_K = gK * n ** 4 * (V - V_K)
        I_L = gL * (V - V_L)
        return -I_Na - I_K - I_L

    @staticmethod
    def _dS(V, n, m, h, t, gNa, V_Na, gK, V_K, gL, V_L):
        dn = (
            0.1 * (1 - 0.1 * V) * (1 - n) / (np.exp(1 - 0.1 * V) - 1)
            - 0.125 * np.exp(-V / 80) * n
        )
        dm = (2.5 - 0.1 * V) * (1 - m) / (np.exp(2.5 - 0.1 * V) - 1) - 4 * np.exp(
            -V / 18
        ) * m
        dh = 0.07 * np.exp(-V / 20) * (1 - h) - h / (np.exp(3 - 0.1 * V) + 1)
        return dn, dm, dh

    @staticmethod
    def _J(ion, V, n, m, h, t, gNa, V_Na, gK, V_K, gL, V_L):

        if ion is Species.Na:
            return (gNa * m ** 3 * h * (V - V_Na)) / 96.48533132838746
        elif ion is Species.K:
            return (gK * n ** 4 * (V - V_K)) / 96.48533132838746
        else:
            return 0 * V

    def S0(self):
        """Return the initial value for the state variable
        n = 0.317
        m = 0.0526
        h = 0.5734
        """
        return 0.317, 0.0526, 0.5734


class Hodgkin_Huxley_red(Channel):
    """Hodgkin Huxley channel reduced version.

    Simplification of the classical HH, with m = m_{\infty}, and h = 0.89 - 1.1 n
    (see Mathematical Physiology, Keener J. and Sneyd J., chap.5)

    Parameters
    ----------
    gNa : float
        [mS/cm2] Conductance of sodium channel
    V_Na : float
        [mV] Equilibrium potential of sodium channel
    gK :  float
        [mS/cm2] Conductance of potasium channel
    V_K : float
        [mV] Equilibrium potential of potasium channel
    gL :  float
        [mS/cm2] conductance of leak channel
    V_L : float
        [mV] Equilibrium potential of leak channel

    """

    nb_var = 1
    param_names = ("gNa", "V_Na", "gK", "V_K", "gL", "V_L")

    def __init__(self, gNa=120, V_Na=115, gK=36, V_K=-12, gL=0.3, V_L=10.6):

        self.params = {
            "gNa": gNa / 100,  # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
            "V_Na": V_Na,
            "gK": gK / 100,  # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
            "V_K": V_K,
            "gL": gL / 100,  # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
            "V_L": V_L,
        }

    @staticmethod
    def _I(V, n, t, gNa, V_Na, gK, V_K, gL, V_L):
        alpha_m = (2.5 - 0.1 * V) / (np.exp(2.5 - 0.1 * V) - 1)
        beta_m = 4 * np.exp(-V / 18)
        m = alpha_m / (alpha_m + beta_m)
        h = 0.89 - 1.1 * n
        I_Na = gNa * m ** 3 * h * (V - V_Na)
        I_K = gK * n ** 4 * (V - V_K)
        I_L = gL * (V - V_L)
        return -I_Na - I_K - I_L

    @staticmethod
    def _dS(V, n, t, gNa, V_Na, gK, V_K, gL, V_L):
        dn = (
            0.1 * (1 - 0.1 * V) * (1 - n) / (np.exp(1 - 0.1 * V) - 1)
            - 0.125 * np.exp(-V / 80) * n
        )
        return dn

    @staticmethod
    def _J(ion, V, n, t, gNa, V_Na, gK, V_K, gL, V_L):
        """
        Return the flux of ion [aM/ms/um2] of the mechanism towards inside
        """
        alpha_m = (2.5 - 0.1 * V) / (np.exp(2.5 - 0.1 * V) - 1)
        beta_m = 4 * np.exp(-V / 18)
        m = alpha_m / (alpha_m + beta_m)
        h = 0.89 - 1.1 * n
        I_Na = gNa * m ** 3 * h * (V - V_Na)
        I_K = gK * n ** 4 * (V - V_K)
        if ion is Species.Na:
            return -I_Na / 96.48533132838746
        elif ion is Species.K:
            return -I_K / 96.48533132838746
        else:
            return 0 * V

    def S0(self):
        """Return the initial value for the state variable

        n = 0.316
        """
        return 0.3162


class Hodgkin_Huxley_Ca(Channel):
    """Ca channel to add to Hodgkin Huxley

    Parameters
    ----------
    gCa : float
        [mS/cm2] Conductance of calcium channel
    V_Ca : float
        [mV] Equilibrium potential of calcium channel

    """

    nb_var = 2
    param_names = ("gCa", "V_Ca")

    def __init__(self, gCa=14.5, V_Ca=115):  #
        self.params = {
            "gCa": gCa / 100,  # conversion mS/cm2 in nS/μm2: 1 mS/cm2 = 0.01 nS/μm2
            "V_Ca": V_Ca,
        }

    @staticmethod
    def _I(V, m, h, t, gCa, V_Ca):
        """
        Return the net surfacic current [pA/um2] of the mechanism towards inside
        """
        I_Ca = gCa * m ** 3 * h * (V - V_Ca)
        return -I_Ca

    @staticmethod
    def _J(ion, V, m, h, t, gCa, V_Ca):
        """
        Return the flux of ion [aM/ms/um2] of the mechanism towards inside
        """
        if ion is Species.Ca:
            return -gCa * m ** 3 * h * (V - V_Ca) / 96.48533132838746 / 2
        else:
            return 0 * V

    @staticmethod
    def _dS(V, m, h, t, gCa, V_Ca):
        dm = 1 / 1.3 * (1 / (1 + np.exp(-V + 102)) - m)
        dh = 1 / 10 * (1 / (1 + np.exp(-V + 24)) - h)
        return dm, dh

    def S0(self):
        """Return the initial value for the state variable

        m = 0
        h = 1
        """
        return 0, 1


class AMPAR(Channel):
    """Point channel with a AMPAr-type current starting at time t0.

    Parameters
    ----------
    t0: float
        start of the current [ms]
    gampa: float
        max conductance of Ampar [nS]
    tampa1: float
        Ampar time constant [ms]
    tampa2: float
        Ampar time constant [ms]
    V_ampa: float
        Ampar Nernst potential [mV]

    """

    param_names = ("gampa", "tampa1", "tampa2", "V_ampa", "t0")

    def __init__(self, t0, gampa=0.02, tampa1=0.3, tampa2=3, V_ampa=70):
        self.params = {
            "t0": t0,
            "gampa": gampa,
            "tampa1": tampa1,
            "tampa2": tampa2,
            "V_ampa": V_ampa,
        }

    @staticmethod
    def _I(V, t, t0, gampa, tampa1, tampa2, V_ampa):
        return (
            ((t <= t0 + 20) & (t >= t0))
            * (-gampa)
            * (1 - np.exp(-np.abs(t - t0) / tampa1))
            * np.exp(-np.abs(t - t0) / tampa2)
            * (V - V_ampa)
        )

    @staticmethod
    def _J(ion, V, t, t0, gampa, tampa1, tampa2, V_ampa):
        """
        Return the flux of ion [aM/ms/um2] of the mechanism towards inside
        """
        if ion is Species.Ca:
            return ((t <= t0 + 20) & (t >= t0)) * np.maximum(
                -gampa
                * (1 - np.exp(-np.abs(t - t0) / tampa1))
                * np.exp(-np.abs(t - t0) / tampa2)
                * (V - V_ampa)
                / 96.48533132838746
                / 2
                * 0.014,
                0,
            )
        else:
            return 0 * V


class NMDAR(Channel):
    """Point channel with a NMDAr-type current starting at time t0.

    Voltage-dependent flow of sodium (Na+) and small amounts of calcium (Ca2+)
    ions into the cell and potassium (K+) out of the cell.

    Parameters
    ----------
    t0: float
        start of the current [ms]
    gnmda: float
        max conductance of NMDAr [nS]
    tnmda1: float
        NMDAr time constant [ms]
    tnmda2: float
        NMDAr time constant [ms]
    V_nmda: float
        NMDAr Nernst potential [mV]

    """

    param_names = ("t0", "gnmda", "tnmda1", "tnmda2", "V_nmda")

    def __init__(self, t0, gnmda=0.2, tnmda1=11.5, tnmda2=0.67, V_nmda=75):

        self.params = {
            "t0": t0,
            "gnmda": gnmda,
            "tnmda1": tnmda1,
            "tnmda2": tnmda2,
            "V_nmda": V_nmda,
        }

    @staticmethod
    def _I(V, t, t0, gnmda, tnmda1, tnmda2, V_nmda):
        return (
            -((t <= t0 + 50) & (t >= t0))
            * gnmda
            * (np.exp(-np.abs(t - t0) / tnmda1) - np.exp(-np.abs(t - t0) / tnmda2))
            / (1 + 0.33 * 2 * np.exp(-0.06 * (V - 65)))
            * (V - V_nmda)
        )

    @staticmethod
    def _J(ion, V, t, t0, gnmda, tnmda1, tnmda2, V_nmda):
        """
        Return the flux of ion [aM/ms/um2] of the mechanism towards inside
        """
        if ion is Species.Ca:
            return ((t <= t0 + 50) & (t >= t0)) * np.maximum(
                -gnmda
                * (np.exp(-np.abs(t - t0) / tnmda1) - np.exp(-np.abs(t - t0) / tnmda2))
                / (1 + 0.33 * 2 * np.exp(-0.06 * (V - 65)))
                * (V - V_nmda)
                / 96.48533132838746
                / 2
                * 0.15,
                0,
            )
        else:
            return 0 * V


def custom(func, name="Custom channel"):
    C = type(name, (Channel,), {"_I": func})
    return C()
