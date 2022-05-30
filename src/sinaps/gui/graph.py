import numpy as np
from scipy import interpolate
import hvplot.pandas
import pandas as pd
import holoviews as hv
from holoviews.operation.datashader import rasterize
import networkx as nx


class Plotter:
    def __init__(self, obj):
        self.obj = obj

    def __call__(self):
        return self.V()

    def V(self, max_plot=10):
        V = self.obj.V.copy()
        return self._plot_lines(V, "potential (mv)", max_plot)

    def C(self, ion, max_plot=10):
        C = self.obj.C(ion).copy()
        return self._plot_lines(C, "concentration (mM)", max_plot)

    def I(self, ch_cls, max_plot=10):
        I = self.obj.I(ch_cls).copy()
        return self._plot_lines(I, "current (pA)", max_plot)

    def _plot_lines(self, df, label, max_plot):
        if issubclass(type(df.columns), pd.MultiIndex):
            df.columns = df.columns.map("{0[0]} - {0[1]}µm".format)
        else:
            df.columns = df.columns.map(("{}".format(self.obj.name) + " - {}µm").format)
        df.index.name = "time (ms)"
        step = max(int(sum([s.nb_comp for s in self.obj]) / max_plot), 1)
        plot = df.loc[::, ::step].hvplot(responsive=True, height=400, ylabel=label)

        return plot


class SimuView:
    def __init__(self, simu):
        self.simu = simu

    def field_plot(self, df, label, time=None, res=None, neuron=True, dynamic=True):
        # res : resolution : #um
        nrn = self.simu.N
        # field plot
        if time is not None:
            df = df[slice(*time)]
        dfi = interpolate.interp1d(
            nrn.indexV_flat(), df.values, fill_value="extrapolate"
        )
        if res is None:
            res = max(max(self.simu.N.indexV_flat()) / 1000, 1)
        y = np.arange(0, max(self.simu.N.indexV_flat()) + res, res)

        p_field = rasterize(
            hv.QuadMesh(
                (df.index.values, y, dfi(y).T), ["Time (ms)", "Position (μm)"], label
            )
        )
        p_field.opts(
            width=600,
            height=600,
            tools=["hover"],
            cmap="fire",
            colorbar=True,
            clabel=label,
        )

        if not neuron:
            return p_field
        else:
            # Neuron representation
            g_2 = nx.Graph()
            edges = nrn.sections.values()
            ej = nrn.__traversal_source__
            branch = 0
            x = 0
            line = []
            path = [0]
            for e in edges:
                if e[0] != ej:
                    line += [
                        {
                            "Position (μm)": path,
                            "h": [0] * len(path),
                            "branch": branch % 20,
                        }
                    ]
                    branch += 1
                    path = [x]

                sec = nrn.graph.edges[e]["section"]
                g_2.add_edge(e[0], e[1], branch=branch, x=x)
                ej = e[1]
                path.append(x + sec.L)
                x += sec.L
            line += [
                {"Position (μm)": path, "h": [0] * len(path), "branch": branch % 20}
            ]

            p_graph = hv.Graph.from_networkx(g_2, nrn.plot.layout)
            p_line = hv.Path(line, ["h", "Position (μm)"], vdims="branch")

            p_graph.opts(
                width=300,
                height=600,
                padding=0.1,
                node_size=0,
                xaxis=None,
                yaxis=None,
                edge_line_width=2,
                edge_color=hv.dim("branch") % 20,
                edge_cmap="Category20b",
                inspection_policy="edges",
                edge_hover_line_color="green",
            )
            p_line.opts(
                color="branch",
                cmap="Category20b",
                line_width=20,
                xaxis=None,
                yaxis=None,
                height=600,
                width=30,
            )

            if not dynamic:
                return p_graph + p_line + p_field
            else:

                # Dynamic selection

                Branch = hv.streams.Stream.define("branch", branch=None)
                branch_stream = Branch()
                # Edge selection
                dmap1 = hv.DynamicMap(
                    lambda branch: p_graph.select(branch=branch),
                    streams=[branch_stream],
                )

                # Field selection
                t0, t1 = df.index[0], df.index[-1]

                def select(branch):
                    if branch == -1 or branch is None:
                        return hv.Polygons(data=None)
                    else:
                        return hv.Polygons(
                            {
                                "y": pos[[branch, branch, branch + 1, branch + 1]],
                                "x": [t0, t1, t1, t0],
                            }
                        )

                dmap2 = hv.DynamicMap(select, streams=[branch_stream])

                # Interactivity with pointer position
                pos = p_graph.data.groupby("branch").min("x")["x"].sort_index()
                pos[pos.index[-1] + 1] = x

                def branch_update(y):
                    if y < 0 or y > x:
                        new_branch = -1
                    else:
                        new_branch = pos.index[pos <= y][-1]
                    if new_branch != branch_stream.branch:
                        branch_stream.event(branch=new_branch)

                pointer1 = hv.streams.PointerY(y=None, source=p_line)
                pointer1.add_subscriber(branch_update)
                pointer2 = hv.streams.PointerY(y=None, source=p_field)
                pointer2.add_subscriber(branch_update)

                # Selection apperance
                dmap1.opts(edge_color="red", edge_line_width=5)
                dmap2.opts(alpha=0.3, color="white")

                return p_graph * dmap1 + p_line + p_field * dmap2

    def V_field(self, **kwargs):
        return self.field_plot(self.simu.V, "Voltage (mV)", **kwargs).opts(
            title="Potential"
        )

    def C_field(self, ion, **kwargs):
        return self.field_plot(self.simu.C[ion], "Concentration (mM/L)", **kwargs).opts(
            title="{} Concentration".format(ion)
        )

    def I_field(self, ch_cls, **kwargs):
        return self.field_plot(
            self.simu.current(ch_cls), "Current (pA)", **kwargs
        ).opts(title="{} Current".format(ch_cls.__name__))

    def __call__(self, **kwargs):
        return self.simu[:].plot(**kwargs)

    def V(self, **kwargs):
        return self.simu[:].plot.V(**kwargs)

    def I(self, ch_cls, **kwargs):
        return self.simu[:].plot.I(ch_cls, **kwargs)

    def C(self, ion, **kwargs):
        return self.simu[:].plot.C(ion, **kwargs)
