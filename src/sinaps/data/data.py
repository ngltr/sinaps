"""Provides tools for working with objects."""
import re


from sinaps.core.model import Section
from sinaps.core.model import Neuron as _Neuron
from sinaps.core.simulation import Simulation as _Simulation

try:
    from sinaps.gui.graph import Plotter
    from sinaps.gui.neuron import NeuronView

    GUI = True
except (ModuleNotFoundError):
    GUI = False


def array(cls_obj, getter=True, setter=True):
    """
    Class decorator to indicate that the class is a list of objects.

    It provides methods for setting and getting attributes of a set of object
    at the same time

    """

    def decorator(cls):
        class ObjectList(cls, list):
            __name__ = cls.__name__

            def _set(self, param, value):
                for s in self:
                    s.__setattr__(param, value)

            def _get(self, param):
                values = [s.__getattribute__(param) for s in self]
                if len(set(values)) == 1:  # All values are identic
                    return values[0]
                else:
                    return values

        fget, fset = None, None
        for p in cls_obj.param:
            if getter:

                def fget(self, p=p):
                    return self._get(p)

            if setter:

                def fset(self, value, p=p):
                    return self._set(p, value)

            type.__setattr__(
                ObjectList, p, property(fget=fget, fset=fset, doc=cls_obj.param[p].doc)
            )
        return ObjectList

    return decorator


@array(Section)
class SectionList:
    def add_channel(self, channel, x=None):
        for s in self:
            s.add_channel(channel, x)

    def clear_channels(self):
        """Clear all channels"""
        for s in self:
            s.clear_channels()


class Neuron(_Neuron):
    __doc__ = _Neuron.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if GUI:
            self.plot = NeuronView(self)

    def __len__(self):
        return len(self.sections)

    def getsections(self, key):
        sec = self[key]
        if not issubclass(type(sec), list):
            sec = SectionList([sec])
        return sec

    @property
    def nodes(self, *args, **kwargs):
        """underlying graph nodes"""
        return self.graph.nodes(*args, **kwargs)

    def add_node_data(self, nodes=None, **kwargs):
        """Add data to graph nodes

        This is useful for plotting
        """
        if nodes is None:
            nodes = self.nodes
        for nd in self.nodes:
            self.graph.add_nodes_from(nodes, **kwargs)

    def __getitem__(self, key):
        def get(key):
            if issubclass(type(key), int):
                return [e["section"] for e in self.graph[key].values()]
            elif issubclass(type(key), slice):
                return list(self.sections)[key]
            elif issubclass(type(key), str):
                return [s for s in self.sections if re.match(key, s.name) is not None]
            else:
                return sum([get(k) for k in key], [])

        sec = get(key)
        if len(sec) == 1:
            return sec[0]
        else:
            return SectionList({*sec})

    def __iter__(self):
        return self.sections.__iter__()

    def leaves(self):
        """Return leaves section"""
        G = self.graph
        return [n for n in G.nodes if G.degree(n) == 1]

    def __repr__(self):
        return "Neuron(name='{obj.name}', {nsec} sections)".format(
            obj=self, nsec=len(self)
        )


@array(Section, setter=False)
class SectionListSimu:
    def __init__(self, list_, simu):
        super().__init__(list_)
        self.simu = simu
        if GUI:
            self.plot = Plotter(self)

    @property
    def V(self):
        return self.simu.V[self.name]

    def C(self, ion):
        return self.simu.C[ion][self.name]

    def I(self, ch_cls):
        return self.simu.current(ch_cls)[self.name]


class Simulation(_Simulation):
    __doc__ = _Simulation.__doc__

    def __len__(self):
        return len(self.N)

    def __getitem__(self, key):
        return SectionListSimu(self.N.getsections(key), self)
