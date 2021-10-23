import numpy as np
from scipy import interpolate
import hvplot.pandas
import pandas as pd
import holoviews as hv
from holoviews.operation.datashader import rasterize


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

    def field(self,df,time=None,res=None): #TODO make it a static method
    # res : resolution : #um
        if time is not None:
            df = df[slice(*time)]
        dfi = interpolate.interp1d(self.simu.N.indexV_flat(), df.values,
                                   fill_value='extrapolate')
        if res is None:
            res = max(max(self.simu.N.indexV_flat())/1000, 1)
        y = np.arange(0, max(self.simu.N.indexV_flat())+res, res)
        return rasterize(
                    hv.QuadMesh((df.index.values, y, dfi(y).T))
            ).opts(xlabel="Time (ms)",
                   ylabel="Position (μm)",
                   width=600,
                   height=600,
                   tools=['hover'])

    def V_field(self,**kwargs):
        return self.field(self.simu.V,**kwargs).opts(title='Potential')

    def C_field(self,ion,**kwargs):
        return self.field(self.simu.C[ion],**kwargs).opts(title='{} Concentration'.format(ion))

    def I_field(self,ch_cls,**kwargs):
        return self.field(self.simu.current(ch_cls),**kwargs).opts(title='{} Current'.format(ch_cls.__name__))

    def __call__(self,**kwargs):
        return self.simu[:].plot(**kwargs)

    def V(self,**kwargs):
        return self.simu[:].plot.V(**kwargs)

    def I(self,ch_cls,**kwargs):
        return self.simu[:].plot.I(ch_cls,**kwargs)

    def C(self,ion,**kwargs):
        return self.simu[:].plot.C(ion,**kwargs)
