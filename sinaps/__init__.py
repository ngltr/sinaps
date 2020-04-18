from sinaps.core.model import Section, Channel
try:
    from sinaps.data.data import Neuron as _Neuron
    from sinaps.data.data import Simulation as _Simulation
    from sinaps.species import Species
    from sinaps.species import INITIAL_CONCENTRATION, DIFFUSION_COEF
    from sinaps.gui.graph import SimuView
    import sinaps.gui.bokeh_surface

    class Neuron(_Neuron):
        __doc__ = _Neuron.__init__.__doc__
        DEFAULT_CONCENTRATION = INITIAL_CONCENTRATION
        DEFAULT_DIFFUSION_COEF = DIFFUSION_COEF

    class Simulation(_Simulation):
        __doc__ = _Simulation.__doc__
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.plot=SimuView(self)
except(ModuleNotFoundError):#nogui
    from sinaps.data.data import Neuron
    from sinaps.data.data import Simulation


import sinaps.channels as channels
import sinaps.io as io

__all__ = ['Section','Channel','Neuron','Simulation','Species','channels','io']
