*************
API Reference
*************

This section provides the complete public API reference for SiNAPS, auto-generated from the docstrings in the project source code.

Section
======================
.. autoclass:: sinaps.Section
   :members: 

Neuron
======================
.. autoclass:: sinaps.Neuron(sections,V_ref)
  :members: add_section, add_sections_from_dict, add_sections_from_list, add_species, add_reaction, leaves, species, nb_nodes, radius_array, nodes, add_node_data

Simulation
======================
.. autoclass:: sinaps.Simulation
   :members: run, run_diff

Channels 
======================
.. autoclass:: sinaps.Channel
   :members: 
.. automodule:: sinaps.channels
   :members: ConstantCurrent, HeavysideCurrent, LeakChannel, Hodgkin_Huxley, Hodgkin_Huxley_Ca, AMPAR, NMDAR
