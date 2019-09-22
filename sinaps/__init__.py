from sinaps.core.model import Section, Ion, Channel
from sinaps.core.model import Neuron as _Neuron
from sinaps.core.simulation import Simulation

from sinaps.gui.graph import NeuronView

import sinaps.channels

class Neuron(_Neuron):
    def __init__(self):
        super().__init__()
        self.view=NeuronView(self)
