import re

import pandas as pd
import numpy as np

from sinaps import Neuron, Section


def read_comp_imaging(filename):
    """Return a Neuron from a Data Comprehensive file"""
    # Read such as nodeId contains list of all tree edges
    # [Edge1_id_node parent,Edge1_id_node_child,Edge2_id_node parent,Edge2_id_node_child,...]
    with open(filename) as f:
        nodeIDs = re.findall(r"NodeID:(-?\d+)\r?\n[PN]", f.read())
    # 2.convert string to int
    nodeIDs = list(map(int, nodeIDs))
    # reaarange values to get list of edges
    edges = [(nodeIDs[k + 1], nodeIDs[k]) for k in range(0, len(nodeIDs) - 1, 2)]

    N = Neuron()
    for parent, child in edges:
        N.add_section(Section(), parent, child)
    N.__traversal_source__ = -1

    return N


def read_swc(filename):
    """Return a neuron from a swc file

    See swc specification at www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

    Parameters
    ----------
    filename : filename or url to swc file

    Examples
    -------
    Create a neuron `nrn` from swc file `neuron.swc`:

    >>> nrn = read_swc('neuron.swc)

    """
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        comment="#",
        index_col=0,
        names=("sample", "structure", "x", "y", "z", "radius", "parent"),
        dtype={
            "sample": int,
            "parent": int,
            "structure": int,
        },
    )

    STRUCT = {
        0: "undefined",
        1: "soma",
        2: "axon",
        3: "dendrite",
        4: "apical_dendrite",
        5: "custom",
    }

    df["structure"] = df["structure"].apply(lambda i: STRUCT[min(i, 5)])

    nrn = Neuron()

    nrn.add_sections_from_dict(
        {
            Section(
                name="{}_{}".format(node.structure, i),
                a=node.radius,
                L=np.sqrt(
                    ((df.loc[i, ["x", "y", "z"]] - df.loc[node.parent]) ** 2).sum()
                ),
            ): (node.parent, i)
            for i, node in df.loc[df["parent"] != -1, :].iterrows()
        }
    )

    nrn.__traversal_source__ = 1
    nrn.plot._layout = {i: np.array([node.x, node.y]) for i, node in df.iterrows()}
    nrn.plot._sections = nrn.sections.copy()

    return nrn
