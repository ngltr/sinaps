.. Sinaps documentation master file, created by
   sphinx-quickstart on Mon Jan 13 17:09:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SiNAPS's documentation!
==================================

`Sinaps` is a Python package providing a fast, flexible and expressive tool to model signal propagation and ionic electrodiffusion in neurons. It is an efficient framework to build computational models of voltage and calcium dynamics. It is based on the Cable theory for voltage propagation, and the Nernst-Plank equation for the electrodiffusion of ions.

.. |pic1| image:: _static/bAP.gif
          :width: 80%

   
.. |pic2| image:: _static/colorbar.png
          :width: 65px

|pic1|  |pic2|

**Figure 1** : `Voltage propagation in neuron triggered by synaptic action potential.`


Contributing
-------

Please ask questions and submit bugs or feature requests on the
`git-hub issue tracker`_ or contact claire.guerrier@univ-cotedazur.fr.

.. _`git-hub issue tracker`: https://github.com/ngltr/sinaps/issues

Credits
-------

`Sinaps` is currently developped in the LJAD, Université Côte d'Azur, Nice, France.


Site map
---------

.. toctree::
    :caption: GETTING STARTED
    :maxdepth: 2

    installing
    conventions
    introduction

.. toctree::
    :caption: TUTORIALS
    :maxdepth: 2

    reaction_diffusion
    complex_geometry
    
.. toctree::
    :caption: IN DEPTH
    :maxdepth: 2

    data_structure
    plotting
    customizing_channels
    model
    api_reference
    GitHub <https://github.com/ngltr/sinaps>


