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

from .bokeh_surface import Surface3d

from sinaps.core.model import Section, Neuron
import sinaps.core.simulation


class Plotter:
    def __init__(self,obj):
        self.obj=obj

    def __call__(self):
        return self.V()


    def V(self,max_plot=10):
        V=self.obj.simu.V[self.obj.name].copy()
        if issubclass(type(V.columns),pd.MultiIndex):
            V.columns=V.columns.map("{0[0]} - {0[1]}µm".format)
        else:
            V.columns=V.columns.map(("{}".format(self.obj.name) +" - {}µm").format)
        step=max(int(sum([s.nb_comp for s in self.obj])/max_plot),1)
        plot = V.loc[::,::step].hvplot(responsive=True,height=400,ylabel='potential (mv)')

        return plot

class View:
    """This class contains all methods to view the results
    of a simulation """

    def __init__(self, simu):
        self.simu = simu

    def voltage():
        pd.Series()






class SimuView:
    def __init__(self, simu):
        self.simu=simu

    def _meshgrid(self,ion,n):

        x=self.simu.N.indexV_flat()
        if ion is None:
            sol=self.simu.sol
        else:
            sol=self.simu.sol_diff[ion]
        y=sol.t
        V=interpolate.interp2d(x,y,
            sol.y[:self.simu.N.nb_comp,:].reshape(-1))
        X = np.arange(x[0], x[-1], (x[-1]-x[0])/n )
        Y = np.arange(y[0], y[-1], (y[-1]-y[0])/n )
        X2, Y2 = np.meshgrid(X, Y)
        Z = V(X,Y)
        return X2,Y2,Z

    def graph2D(self,figsize=(10,10),ion=None,**kwargs):
        X,Y,Z = self._meshgrid(ion,1000)
        fig = plt.figure(figsize=figsize,**kwargs)
        fig.gca().matshow(Z.T)

    def V(self,section=np.s_[:],max_plot=10,height=400,**kwargs):
        sec = self.simu.N[section]
        V=self.simu.V[sec.name].copy()
        if issubclass(type(V.columns),pd.MultiIndex):
            sections = sec
            V.columns=V.columns.map("{0[0]} - {0[1]}µm".format)
        else:
            sections = [sec]
            V.columns=V.columns.map(("{}".format(sec.name) +" - {}µm").format)
        step=max(int(sum([s.nb_comp for s in sections])/max_plot),1)
        plot = V.loc[::,::step].hvplot(responsive=True,height=400,ylabel='potential (mv)')
        if type(section) is int:
            plot.label = "Section {}".format(section)
        return plot


    def graph3D(self,ion=None):
        X,Y,Z = self._meshgrid(ion,100)
        source = ColumnDataSource(data=dict(x=X/1000, y=Y, z=Z))
        surface = Surface3d(x="x", y="y", z="z",
        data_source=source, width=1000, height=1000)
        output_file('3d.html')
        show(surface)
        output_notebook

class NeuronView:
    def __init__(self, N):
        self.N=N
        self._x=None
        self._y=None

    def graph(self):
        plt.scatter(self.x, self.y, marker='|')
        for s,(i,j) in self.N.sections.items():
                plt.plot([self.x[i],self.x[j]],[self.y[i],self.y[j]],
                        linewidth=s.a*2,
                        color='grey')


    @property
    def x(self):
        if self._x is None:
            self._create_coordinate()
        return self._x

    @property
    def y(self):
        if self._y is None:
            self._create_coordinate()
        return self._y

    def _create_coordinate(self,
                      permut= np.random.permutation,
                      start_angle=0,
                      diff_angle=math.pi/6,
                     ):
        """create the nodes coordinate from tree structure for ploting the neuron"""

        n=self.N.nb_nodes
        mat=self.N.adj_mat
        self._x=[0]*n
        self._y=[0]*n

        def set_coordinate_from_node(i,angle_i):
            nb_children=np.count_nonzero(mat[i])
            k=-(nb_children-1)/2.0
            for j in permut(n):
                if mat[i,j]:
                    angle_j=angle_i+k*diff_angle
                    k=k+1
                    self._x[j]=self._x[i]+math.cos(angle_j)*mat[i,j].L
                    self._y[j]=self._y[i]-math.sin(angle_j)*mat[i,j].L
                    set_coordinate_from_node(j,angle_j)

        set_coordinate_from_node(0,start_angle)
