import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import cm
from scipy import interpolate
from bokeh.models import ColumnDataSource
from bokeh.io import show, output_notebook, output_file
import warnings
import hvplot.pandas
import pandas as pd
import holoviews as hv
from holoviews.operation.datashader import rasterize

from sinaps.core.model import Section, Neuron
import sinaps.core.simulation


class Plotter:
    def __init__(self,obj):
        self.obj=obj

    def __call__(self):
        return self.V()


    def V(self,max_plot=10):
        V=self.obj.V.copy()
        return self._plot_lines(V,'potential (mv)',max_plot)


    def C(self,ion,max_plot=10):
        C=self.obj.C(ion).copy()
        return self._plot_lines(C,'concentration (mM)',max_plot)

    def I(self,ch_cls,max_plot=10):
        I=self.obj.I(ch_cls).copy()
        return self._plot_lines(I,'current (pA)',max_plot)


    def _plot_lines(self,df,label,max_plot):
        if issubclass(type(df.columns),pd.MultiIndex):
            df.columns=df.columns.map("{0[0]} - {0[1]}µm".format)
        else:
            df.columns=df.columns.map(("{}".format(self.obj.name) +" - {}µm").format)
        df.index.name='time (ms)'
        step=max(int(sum([s.nb_comp for s in self.obj])/max_plot),1)
        plot = df.loc[::,::step].hvplot(responsive=True,height=400,ylabel=label)

        return plot



class SimuView:
    def __init__(self, simu):
        self.simu=simu

    def field(self,df): #TODO make it a static method
        return rasterize(hv.QuadMesh(
        (df.index.values,self.simu.N.indexV_flat(),df.values.T))
        ).opts(xlabel="Time (ms)",ylabel="Position (μm)",width=600,height=600)

    def V_field(self):
        return self.field(self.simu.V).opts(title='Potential')

    def C_field(self,ion):
        return self.field(self.simu.C[ion]).opts(title='{} Concentration'.format(ion))

    def I_field(self,ch_cls):
        return self.field(self.simu.current(ch_cls)).opts(title='{} Current'.format(ch_cls.__name__))

    def __call__(self,**kwargs):
        return self.simu[:].plot(**kwargs)

    def V(self,**kwargs):
        return self.simu[:].plot.V(**kwargs)

    def I(self,ch_cls,**kwargs):
        return self.simu[:].plot.I(ch_cls,**kwargs)

    def C(self,ion,**kwargs):
        return self.simu[:].plot.C(ion,**kwargs)
