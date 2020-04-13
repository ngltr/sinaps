import networkx as nx
import holoviews as hv


class NeuronView:
    def __init__(self, nrn):
        self.nrn = nrn
        self._layout=None
        self._sections=None

    def __call__(self,*args,**kwargs):
        return self.graph(*args,**kwargs)

    def graph(self,layout=None):
        """Plot the graph of the neuron"""
        if layout is None:
            layout = self.layout()

        G = self.nrn.graph
        #Convert section to attributes for hover tool
        G2 = nx.Graph()
        for e in G.edges:
            s= G.edges[e]['section']
            G2.add_edge(e[0],e[1],
                       name=s.name,
                       L=s.L,
                       a=s.a,
                       C_m=s.C_m,
                       R_l=s.R_l
                       )
        G2.add_nodes_from(G.nodes.data())
        plot = hv.Graph.from_networkx(G2,layout)

        plot.opts(width=500,height=500,
                  xaxis = None,
                  yaxis = None,
                  padding=0.1,
                  node_size=2,
                  edge_line_width=hv.dim('a'),
                  edge_color='blue',
                  inspection_policy='edges',
                  edge_hover_line_color='green')

        return plot

    def layout(self,layout=nx.kamada_kawai_layout,force=False):
        """Layout of the neuron, recalculate if neuron structure has changed, use cached version if no"""
        if self._layout is None or (self._sections != self.nrn.sections) or force:
            print("Calculating layout...",end="")
            self._sections = self.nrn.sections.copy()
            G = self.nrn.graph
            #Convert section to attributes for hover tool
            G2 = nx.Graph()
            for e in G.edges:
                s= G.edges[e]['section']
                G2.add_edge(e[0],e[1],L=s.L)
            self._layout = layout(G2, weight='L')
            print("[OK]")
        return self._layout
