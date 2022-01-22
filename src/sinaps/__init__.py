from .core.model import Section, Channel, VoltageSource

__version__ = "0.3.0"
__all__ = [
    "Section",
    "Channel",
    "Neuron",
    "VoltageSource",
    "Simulation",
    "Species",
    "channels",
    "io",
]

try:
    from .data.data import Neuron as _Neuron
    from .data.data import Simulation as _Simulation
    from .core.species import Species
    from .core.species import INITIAL_CONCENTRATION, DIFFUSION_COEF
    from .gui.graph import SimuView

    class Neuron(_Neuron):
        """Neuron."""

        __doc__ = _Neuron.__doc__
        DEFAULT_CONCENTRATION = INITIAL_CONCENTRATION
        DEFAULT_DIFFUSION_COEF = DIFFUSION_COEF

    class Simulation(_Simulation):
        """Simulation with plotter object."""

        __doc__ = _Simulation.__doc__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.plot = SimuView(self)

except ModuleNotFoundError:  # nogui
    from .data.data import Neuron
    from .data.data import Simulation

from .core import channels
from .data import io
