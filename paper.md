---
title: 'Sinaps: A Python library to simulate voltage dynamics and ionic electrodiffusion in neurons'
tags:
  - Python
  - neuroscience
  - voltage propagation
  - calcium electrodiffusion
  - cable theory
  - Nernst-Planck equations
authors:
  - name: Nicolas Galtier
    affiliation: "2"
  - name: Claire Guerrier
    orcid: 0000-0002-1128-1069 
    affiliation: "1" #
affiliations:
 - name: CNRS, LJAD, Université Côte d'Azur
   index: 1
 - name: Independent Researcher
   index: 2
date: 1st Novembre 2021
bibliography: paper.bib
---

# Summary

How do neuronal dendrites collect, process and transmit information? What is the role of neurons'
specific geometry on neuronal activity? The old postulate that dendrites serve mainly to connect
neurons and to convey information, with no specific role in cognitive processes, is currently
challenged by the emergence of novel experimental techniques [@London:2005 ; @Basak:2018]. Hence, the 
role of the dendritic tree in transforming synaptic input into neuronal output is now a leading
question in developmental neuroscience. In particular, using genetically-encoded Ca$^{2+}$-
indicators, state-of-the-art techniques have been developed to track the calcium dynamics within
the entire dendritic tree, at high spacial and temporal scales [@Sakaki:2020]. Tracking neuronal
activity through calcium dynamics is nevertheless ambitious. Calcium concentration fluctuations
are known to reflect neuronal activity in a very complex way, as ions can flow from many sources
that interact non-linearly with each other. There is thus an enhanced need for modeling to
resolve what can be seen as an inverse problem: finding from experimental recordings the markers
of synaptic activity, and distinguish in the calcium signals the different calcium sources. 

Here, we present `Sinaps`, which is an easy-to-use Python library to simulate voltage propagation, ionic electrodiffusion and chemical reactions in neurons. 

![Left: a complete neuronal geometry created using `Sinaps`. Right: Simulation of voltage propagation in the dendritic tree represented in Left. The color code in the middle maps the structure of the geometry on the left, to the vertical position on the right.\label{fig:volt}](Fig1.png)

# Statement of need

Numerous softwares have been designed to realize simulations of voltage dynamics and ion concentration in neurons [@Bower:1998 ; @Carnevale:2006]. Most of those softwares are using the Cable theory model, and are designed toward neuronal network simulation. Models for ion concentration dynamics in neurons, based on the Nernst-Planck model coupled to specific equations describing the potential, are also available [@Saetra:2020 ; @Solbra:2018]. 

Among these models, the `Sinaps` library is designed to realize the fast computation of voltage propagation and ionic electrodiffusion in complex neuronal geometries, at a fine spatial scale. It is based on the Cable equation for voltage propagation, coupled to a one-dimensional Nernst-Planck equation for ionic electrodiffusion. The library can be used to build custom set-ups, with easy access to all the variables. Finally, the class structure renders the code easily editable.

The library has been designed for neuroscience laboratories using both an experimental and a modeling approach. It includes the code to simulate voltage dynamics and ionic electrodiffusion, Hodgkin-Huxley type membrane channels, and chemical reactions. Templates to code custom reaction-diffusion mechanisms, as well as specific membrane channels are provided. We also provide the possibility to load a full morphometric geometry from data following [neuromopho.org](http://neuromorpho.org/) file type (swc file) \autoref{fig:volt}. Hence, our library provides an easy way to simulate voltage and ionic dynamics, at the spacial scale reached by morphometric techniques, and at a temporal scales not yet available for \textit{in vivo} imaging of the full neuronal scale.

# Acknowledgements

This work was supported by the Fyssen foundation.

# References
