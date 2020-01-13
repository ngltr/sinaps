from sinaps.core.model import Section, Channel
from sinaps.core.model import Neuron as _Neuron
from sinaps.core.simulation import Simulation as _Simulation
from sinaps.species import Species
from sinaps.species import INITIAL_CONCENTRATION, DIFFUSION_COEF
from sinaps.gui.graph import NeuronView, SimuView

import sinaps.gui.bokeh_surface

import sinaps.channels as channels

class Neuron(_Neuron):
    def __init__(self):
        super().__init__()
        self.view=NeuronView(self)

    def add_species(self,species,C0=None,D=None):
        if C0 is None:
            C0 = INITIAL_CONCENTRATION
        if D is None:
            D = DIFFUSION_COEF
        super().add_species(species,C0,D)

class Simulation(_Simulation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.view=SimuView(self)

try:
    """Sortie des graphiques dans jupyter notebook """
    get_ipython().magic('matplotlib inline')
    from bokeh.io import output_notebook
    output_notebook()
    import pandas
    pandas.set_option('display.max_rows', 500)
    pd.set_option('plotting.backend', 'pandas_bokeh')
    pd.plotting.output_notebook()
except:
    pass
