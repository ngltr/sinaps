# -*- coding: utf-8 -*-
import pytest

from sinaps.core.model import Neuron, Section
from sinaps.core.simulation import Simulation
import sinaps.core.channels as ch


CHANNEL_LIST = [
    ch.ConstantCurrent(5),
    ch.AMPAR(1),
    ch.NMDAR(1),
    ch.HeavysideCurrent(5, 1, 2),
    ch.Hodgkin_Huxley(),
    ch.Hodgkin_Huxley_Ca(),
    ch.LeakChannel(),
]


@pytest.fixture(params=CHANNEL_LIST)
def neuron_with_channel():
    sec = Section()
    sec.add_channel(ch.ConstantCurrent(5))
    nrn = Neuron({sec:(0, 1)})
    return nrn


def test_channels_I(neuron_with_channel):
    sim = Simulation(neuron_with_channel, dx=10)
    sim.run((0, 5))

