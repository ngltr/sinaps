# -*- coding: utf-8 -*-
import pytest

from sinaps.core.model import Section, Neuron, VoltageSource
from sinaps.core.channels import ConstantCurrent
from sinaps.core.species import Species
import numpy as np


@pytest.fixture()
def channel():
    return ConstantCurrent(1)


@pytest.fixture()
def section():
    return Section()


@pytest.fixture()
def section_with_channel(channel):
    sec = Section()
    sec.add_channel(channel)
    return sec


@pytest.fixture()
def neuron():
    nrn = Neuron([(0, 1), (1, 2), (1, 3), (3, 4)])
    return nrn


def test_section_clear_channels(section_with_channel):
    section_with_channel.clear_channels()
    assert len(section_with_channel.channels) == 0


def test_section_add_channel(section, channel):
    section.add_channel(channel)
    assert len(section.channels) == 1
    section.add_channel(channel, 1)
    assert len(section.channels) == 2
    assert section.channels[1].position == 1


def test_section_add_voltage_source(section):
    c = VoltageSource(lambda x: 1)
    s = Section()
    s.add_voltage_source(c)
    assert len(s.Vsource) == 1
    s.add_voltage_source(c, 0.23)
    assert s.Vsource[1].position == 0.23


def test_section_c_m():
    sec = Section(C_m=22, a=3)  # uF/cm2
    assert sec.c_m == pytest.approx(np.pi * 2 * 3 * 22 / 100)  # pF/um2


def test_section_r_l():
    sec = Section(R_l=138, a=3)
    assert sec.r_l == pytest.approx(138 / 1e5 / (np.pi * 3 * 3))
