from sinaps.core.model import Section, Ion, Channel
from sinaps.core.model import Neuron as _Neuron
from sinaps.core.simulation import Simulation as _Simulation
import sinaps.ions as ions

from sinaps.gui.graph import NeuronView, SimuView

import sinaps.gui.bokeh_surface

import sinaps.channels as channels

class Neuron(_Neuron):
    def __init__(self):
        super().__init__()
        self.view=NeuronView(self)

class Simulation(_Simulation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.view=SimuView(self)


import pandas as pd
pd.set_option('plotting.backend', 'pandas_bokeh')
pd.plotting.output_notebook()
