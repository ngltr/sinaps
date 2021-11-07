# -*- coding: utf-8 -*-
import pytest

from sinaps.core.simulation import Simulation
from sinaps.core.model import Section, Neuron
from sinaps.core.species import Species


@pytest.fixture()
def neuron():
    nrn = Neuron([(0, 1), (1, 2), (1, 3), (3, 4)])
    return nrn


def test_simulation_run(neuron):
    sim = Simulation(neuron, dx=10)
    sim.run((0, 12))
    assert sim.V.index.max() == 12


def test_simulation_run_diff(neuron):
    neuron.add_species(Species.Ca, C0=2, D=1)
    sim = Simulation(neuron, dx=10)
    sim.run((0, 12))
    sim.run_diff()
    assert sim.C[Species.Ca].index.max() == 12
