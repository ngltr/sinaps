import sinaps as sn
import numpy as np


def test_HH():
    nrn = sn.Neuron()
    sec = sn.Section()
    sec.add_channel(sn.channels.Hodgkin_Huxley())
    sec.add_channel(sn.channels.PulseCurrent(100, 2, 4), 0)
    nrn.add_section(sec, 0, 1)
    assert nrn[0] == sec
    assert nrn[1] == sec
    sim = sn.Simulation(nrn, dx=10)
    sim.run((0, 20))

    assert sim.V.max().min() > 100


def test_voltage_source():
    nrn = sn.Neuron()
    sec = sn.Section(L=1000)
    sec.add_voltage_source(sn.VoltageSource(lambda t: np.sin(t)))
    nrn.add_section(sec, 0, 1)
    sim = sn.Simulation(nrn, 10)
    sim.run((0, 10))

    assert sim.V.iloc[:, 1].max() > 0.6
