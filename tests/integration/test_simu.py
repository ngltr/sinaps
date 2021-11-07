import sinaps as sn


def test_HH():
    nrn = sn.Neuron()
    sec = sn.Section()
    sec.add_channel(sn.channels.Hodgkin_Huxley())
    sec.add_channel(sn.channels.HeavysideCurrent(100, 2, 4), 0)
    nrn.add_section(sec, 0, 1)
    assert nrn[0] == sec
    assert nrn[1] == sec
    sim = sn.Simulation(nrn, dx=10)
    sim.run((0, 20))

    assert sim.V.max().min() > 110
