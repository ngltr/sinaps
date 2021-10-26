# -*- coding: utf-8 -*-
from sinaps import Neuron


def test_neuron_repr():
    nrn = Neuron()
    assert(isinstance(nrn.__repr__(), str))
    assert(len(nrn.__repr__())) > 0





#%%

if __name__ == '__main__':
   test_neuron_repr() 