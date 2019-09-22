import matplotlib.pyplot as plt
import math
import numpy as np


class View:
    """This class contains all methods to view the results
    of a simulation """

    def __init__(self, simu):
        self.simu = simu

    def voltage():
        pd.Series()



class NeuronView:
    def __init__(self, N):
        self.N=N
        self._x=None
        self._y=None

    def graph(self):
        plt.scatter(self.x, self.y, marker='|')
        for s in self.N.sections:
                plt.plot([self.x[s['i']],self.x[s['j']]],[self.y[s['i']],self.y[s['j']]],
                        linewidth=s['obj'].a*2,
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
