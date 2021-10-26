"""Contains class and methos to plot the neuron structure."""

# %%Imports


import networkx as nx
import holoviews as hv


# %%Class


class NeuronView:
    """Viewer class that contains methods to plot the neuron structure.

    Attributes
    ----------
    nrn : sinaps.Neuron
        Neuron object

    """

    def __init__(self, nrn):
        self.nrn = nrn
        self._layout = None
        self._sections = None

    def graph(self, layout=None):
        """Plot the graph of the neuron.

        Parameters
        ----------
        layout : dict, optional
            A dictionary of positions keyed by node. If None, use a node
            positioning algorithm (See layout).

        Returns
        -------
        holoviews.Graph
            Graph of the neuron

        """
        graph = self.nrn.graph
        if len(graph) <= 2:
            layout = {n: (n, 0) for n in graph}
        if layout is None:
            layout = self.layout()

        # Convert section to attributes for hover tool
        g_2 = nx.Graph()
        for edg in graph.edges:
            sec = graph.edges[edg]["section"]
            g_2.add_edge(
                edg[0],
                edg[1],
                name=sec.name,
                L=sec.L,
                a=sec.a,
                C_m=sec.C_m,
                R_l=sec.R_l,
            )
        g_2.add_nodes_from(graph.nodes.data())
        plot = hv.Graph.from_networkx(g_2, layout)

        plot.opts(
            width=500,
            height=500,
            xaxis=None,
            yaxis=None,
            padding=0.1,
            node_size=2,
            edge_line_width=hv.dim("a"),
            edge_color="blue",
            inspection_policy="edges",
            edge_hover_line_color="green",
        )
        return plot

    def __call__(self, *args, **kwargs):
        """See graph doc."""
        return self.graph(*args, **kwargs)

    __call__.__doc__ = graph.__doc__

    def layout(self, layout=nx.kamada_kawai_layout, force=False):
        """Position nodes of the neuron for graph drawing.

        The result is cached, and recalculated only if neuron structure has
        changed.

        Parameters
        ----------
        layout : method, optional
            Node positioning algorithms for graph drawing.
            The default is networkx.kamada_kawai_layout.
        force : Boolean, optional
            If True, the layout will be recalculated even if it exists.
            The default is False.

        Returns
        -------
        pos: dict
            A dictionary of positions keyed by node.

        """
        if self._layout is None or (self._sections != self.nrn.sections) or force:
            print("Calculating layout...", end="")
            self._sections = self.nrn.sections.copy()
            graph = self.nrn.graph
            # Convert section to attributes for hover tool
            g_2 = nx.Graph()

            for edg in graph.edges:
                sec = graph.edges[edg]["section"]
                g_2.add_edge(edg[0], edg[1], L=sec.L)
            self._layout = layout(g_2, weight="L")
            print("[OK]")

        return self._layout
