#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 09:37:11 2020

@author: guerrier
"""
import holoviews as hv

import sinaps as sn

nrn = sn.Neuron()
sec = sn.Section()
sec.add_channel(sn.channels.PulseCurrent(100, 2, 4), 0)
nrn.add_section(sec, 0, 1)
nrn.add_section(sn.Section(), 1, 2)
sim = sn.Simulation(nrn, dx=10)
sim.run((0, 10))


def assert_graph(obj):
    assert (
        issubclass(type(obj), hv.Element)
        or issubclass(type(obj), hv.Overlay)
        or issubclass(type(obj), hv.NdOverlay)
        or issubclass(type(obj), hv.Layout)
        or issubclass(type(obj), hv.DynamicMap)
    )


#%%


def test_plot_neuron():
    assert_graph(nrn.plot())


#%%


def test_plot_V():
    assert_graph(sim.plot())
    assert_graph(sim.plot.V())


def test_plot_fiels():
    assert_graph(sim.plot.V_field())
    assert_graph(sim.plot.I_field(sn.channels.PulseCurrent))
