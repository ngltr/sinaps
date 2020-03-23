"""
Provides tools for working with objects

"""
import re


from sinaps.core.model import Section
from sinaps.core.model import Neuron as _Neuron
from sinaps.core.simulation import Simulation as _Simulation
from sinaps.gui.graph import Plotter

def array(cls_obj,getter=True,setter=True):
    """
    Class decorator to indicate that the class is a list of objects
    It provides methods for setting and getting attributes of a set of object at the same time

    """
    def decorator(cls):
        class ObjectList(cls,list):
            __name__=cls.__name__
            def _set(self,param,value):
                for s in self:
                    s.__setattr__(param,value)
            def _get(self,param):
                values=[s.__getattribute__(param) for s in self]
                if len(set(values))==1: #All values are identic
                    return values[0]
                else:
                    return values
        fget, fset = None, None
        for p in cls_obj.param:
            if getter: fget=lambda self,p=p:self._get(p)
            if setter: fset=lambda self,value,p=p:self._set(p,value)
            type.__setattr__(ObjectList,p,property(
                fget=fget, fset=fset,
                doc = cls_obj.param[p].doc))
        return ObjectList
    return decorator



@array(Section)
class SectionList:
    def add_channel(self,c):
        for s in self:
            s.add_channel(c)


class Neuron(_Neuron):
    def __len__(self):
        return len(self.sections)

    def getsections(self,key):
        sec = self[key]
        if not issubclass(type(sec), list):
            sec = SectionList([sec])
        return sec


    def __getitem__(self,key):
        if issubclass(type(key), int):
            return list(self.sections)[key]
        elif issubclass(type(key), slice):
            sec = list(self.sections)[key]
        else:
            sec=[s for s in self.sections if re.match(key,s.name) is not None]

        if len(sec) == 1:
            return sec[0]
        else:
            return SectionList(sec)



@array(Section,setter=False)
class SectionListSimu:
    def __init__(self,list_,simu):
        super().__init__(list_)
        self.simu = simu
        self.plot = Plotter(self)
    @property
    def V(self):
        return self.simu.V[self.name]


class Simulation(_Simulation):
    

    def __len__(self):
        return len(self.N)

    def __getitem__(self,key):
        return SectionListSimu(self.N.getsections(key),self)
