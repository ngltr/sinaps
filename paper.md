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
        



<!-- The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).-->

# Statement of need

In this paper, we present 'Sinaps', an easy-to-use and freely available python library to simulate voltage propagation, ionic electrodiffusion and chemical reactions in neurons. This library has been designed for neuroscience laboratories using both an experimental and a modeling approach. It includes the code to simulate voltage dynamic and ionic electrodiffusion, Hodgkin-Huxley type membrane channels, and as well as chemical reaction. Templates to code custom reaction-diffusion mechanisms, as well as specific membrane channels are provided. The detailed geometry of a specific cell can be easily incorporated, or loaded from data following \href{http://neuromorpho.org/}{\color{urlblue}{neuromopho.org}} file type, or from the Allen institute Whole Brain project.

Numerous softwares has been designed to realize such simulations \cite{Neuron,Genesis,Blue brain project,AllenInstitute}. While most of those softwares are using the simple Cable theory model, and are designed toward neuronal networks simulation, our Python library is designed to realize fast simulation of both voltage and ionic dynamics, taking into account electrodiffusion of ions at a fine spatial scale. We also choose to realize the code in Python, which has the advantage of having a code fully transparent with easy access to all the variables. The class structure renders the code easily editable. We also provide the possibility to load a full morphometric geometry from data following \href{http://neuromorpho.org/}{\color{urlblue}{neuromopho.org}} file type, and from the Whole Brain project in the Allen institute. Hence, our library provides an easy way to simulate voltage and ionic dynamics, at the spacial scale reached by morphometric techniques, and at a temporal scales not yet available for \textit{in vivo} imaging of the full neuronal scale.\\


<!--`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.-->

<!-- # Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

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

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }-->

# Acknowledgements

We acknowledge contributions from .

# References
