---
title: 'Sinaps: A Python library to simulate voltage dynamic and ionic electrodiffusion in neurons'
tags:
  - Python
  - neuroscience
  - voltage propagation
  - calcium electrodiffusion
  - cable theory
  - Nernst-Planck equations
authors:
  - name: Nicolas Galtier # note this makes a footnote saying 'co-first author'
    affiliation: Independent researcher
  - name: Claire Guerrier  #^[co-first author] note this makes a footnote saying 'co-first author'
    orcid: XCCC
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Université Côte d'Azur
   index: 1
 - name: Independent Researcher
   index: 2
date: 28 October 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

How neuronal dendrites collect, process and transmit information? What is the role of neurons specific
geometry on neuronal activity? The old postulate that dendrites serve mainly to connect neurons and to
convey information with no specific role in synaptic plasticity is currently challenged by the
emergence of novel experimental techniques. This allowed us to discover its involvement in cellular learning rules, and to challenge
the old postulate that dendrites serve mainly to connect neurons and to convey information with no
specific role in synaptic plasticity \cite{London05,Basak18}. Hence, the role of the dendritic
tree in transforming synaptic input into neuronal output and in defining the relationships
between active synapses is now a leading question in developmental neuroscience. In particular, through genetically-encoded Ca$^{2+}$-indicators, 
the activity of the full tree.  Tracking neuronal
activity through calcium flourescence is nevertheless ambitious. Indeed, calcium concentration fluctuations are known to reflect
neuronal activity in a very complex way, as ions can flow from many sources that interact non-linearly
with each other. There is thus an enhanced need for modeling to resolve what can be seen as an inverse
problem: finding from experimental recordings the active synapses, and differentiate between several
calcium sources. 
        

# Statement of need

`Sinaps` is an easy-to-use and freely available python library to simulate voltage propagation, ionic electrodiffusion and chemical reactions in neurons. This library has been designed for neuroscience laboratories using both an experimental and a modeling approach. It includes the code to simulate voltage dynamic and ionic electrodiffusion, Hodgkin-Huxley type membrane channels, and as well as chemical reaction. Templates to code custom reaction-diffusion mechanisms, as well as specific membrane channels are provided. The detailed geometry of a specific cell can be easily incorporated, or loaded from data following \href{http://neuromorpho.org/}{\color{urlblue}{neuromopho.org}} file type, or from the Allen institute Whole Brain project.

Numerous softwares has been designed to realize such simulations \cite{Neuron,Genesis,Blue brain project,AllenInstitute}. While most of those softwares are using the simple Cable theory model, and are designed toward neuronal networks simulation, our Python library is designed to realize fast simulation of both voltage and ionic dynamics, taking into account electrodiffusion of ions at a fine spatial scale. We also choose to realize the code in Python, which has the advantage of having a code fully transparent with easy access to all the variables. The class structure renders the code easily editable. We also provide the possibility to load a full morphometric geometry from data following \href{http://neuromorpho.org/}{\color{urlblue}{neuromopho.org}} file type, and from the Whole Brain project in the Allen institute. Hence, our library provides an easy way to simulate voltage and ionic dynamics, at the spacial scale reached by morphometric techniques, and at a temporal scales not yet available for \textit{in vivo} imaging of the full neuronal scale.\\

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

# Acknowledgements

We acknowledge contributions from.

# References
