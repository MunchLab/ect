---
title: 'ECT: A Python Package for the Euler Characteristic Transform'
tags:
  - Python
  - Topological Data Analysis
  - Euler Characteristic 
authors:
  - name: Elizabeth Munch
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
    orcid: 0000-0002-9459-9493
  - name: Yemeen Ayub
    affiliation: 1
    orcid: 0000-0000-0000-0000
  - name: Dan Chitwood 
    affiliation: 1
    orcid: 0000-0000-0000-0000
  - name: Sarah McGuire
    affiliation: 2
    orcid: 0000-0000-0000-0000
affiliations:
 - name: Michigan State University, East Lansing, MI, USA
   index: 1
 - name: Pacific Northwest National Lab (PNNL), USA
   index: 2
date: Jan 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishings
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The field of Topological Data Analysis [@Dey2021;@Wasserman2018;@Ghrist2014;@Munch2017] encodes the shape of data in quantifiable representations of the information, sometimes called "topological signatures" or "topological summaries". The goal is to ensure that these summaries are robust to noise and useful in practice. In many methods, richer representations bring higher computation cost, creating a tension between robustness and speed. The Euler Characteristic Transform (ECT) [@Turner2014;@Munch2025] has gained popularity for encoding the information of embedded shapes in $\mathbb{R}^d$---such as graphs, simplicial complexes, and meshes--because it strikes this balance by providing a complete topological summary, yet is typically much faster to compute than its widely used cousin, the Persistent Homology Transform [@Turner2014].

The `ect` Python package offers a fast and well-documented implementation of ECT for inputs in any embedding dimension and with a wide range of complex types. With a few lines of code, users can generate ECT features by sampling directions, computing Euler characteristic curves, and vectorizing them for downstream tasks such as classification or regression. The package includes practical options for direction sampling, normalization, and visualizing various versions of the ECT. These options allow for smooth integration into other scientific package such as `Numpy`, `Scipy`, and `PyTorch`. By lowering the barrier to computing the ECT on embedded complexes, `ect` makes these topological summaries accessible to a wider range of practitioners and domain scientists.

## The Euler Characteristic Transform

We give a high level introduction of the ECT here as defined in [@Turner2014], and direct the reader to [@Munch2025] for a full survey article specifically on the subject. Further, note that the code is built to handle embedded cell complexes of arbitrary dimension, but for ease of introduction, we explain the basics using embedded graphs.

To start, we assume our input is an undirected graph $G$ with a straight-line embedding in 2D given by a map on the vertices $f: V(G) \to \mathbb{R}^2$. A graph can be constructed as seen in \autoref{fig:example_graph}.


![A filtration of a graph showing the sublevel sets of $g_\omega$ for a fixed direction $\omega$. The vertices and edges are added to the filtration as the height increases.\label{fig:filtration}](figures/filtration.png)

<!-- ![Testing scaling](figures/CombineGraphExample.png){ width=20% } -->


For a choice of direction $\theta \in [0,2\pi]$, we can induce a function on the vertex set. 
Thinking of this as $\omega \in \mathbb{S}^1$ by defining the unit vector $\omega = (\cos(\theta), \sin(\theta))$, the function $g_\omega$ is defined on the vertices of $G$ by taking the dot product of the embedding coordinates with the unit vector, specifically
$$
g_\omega(v) = \langle f(v), \omega\rangle.
$$
<!-- This is done in the code using the `g_omega` method as shown.  -->
Some examples are shown in \autoref{fig:example_graph}. 

Now we can set up the ECT for the embedded graph. The ECT is defined as 
$$
\begin{matrix}
\text{ECT}(G): & \mathbb{S}^1 & \to & \text{Func}(\mathbb{R}, \mathbb{Z})\\
& \omega & \mapsto & \{ a \mapsto \chi(g_\omega^{-1}(-\infty,a]) \}
\end{matrix}
$$
Perhaps a better way of looking at this same function for visualization purposes is to treat this function as defined on a cylinder,
$$
\begin{matrix}
\text{ECT}(G): & \mathbb{S}^1 \times \mathbb{R} & \to &  \mathbb{Z}\\
& (\omega,a) & \mapsto & \chi(g_\omega^{-1}(-\infty,a]).
\end{matrix}
$$
After discretizing, the example embedded graph has an ECT matrix as shown in the bottom row of \autoref{fig:example_graph}.
The main functionality of the `ECT` package is to be able to compute the ECT matrix for graphs embedded in $\mathbb{R}^d$ for $d \in \{2,3\}$.

![(Top row) An example of an embedded graph with two choices of function $f_\omega$ drawn as the coloring on the nodes. (Bottom) The ECT matrix of the graph shown.\label{fig:example_graph}](figures/CombineGraphExample.png)


## Extension to higher dimensional embedding

In theory, the ECT can be defined for a space embedded in $\mathbb{R}^d$ for any $d$. 
In practice, for applications geared toward encoding shapes seen in the physical world, this is largely limited to the cases $d=2$ or $d=3$. 
Still, issues arise in applications in the case of $d=3$ where the choice for discretizing the directions chosen on the sphere $\mathbb{S}^2$ is not canonical like the case of $d=2$ and the circle $\mathbb{S}^1$ [@Mardia1999]. 
To this end we have implemented **Make sure this is true** the 3D ECT for graph inputs. 
In order to handle issues with choices of direction discretiziations, we have implemented multiple options for sampling, **Which ones? **
- Uniform in $\theta$ and $\rho$? 
- Whatever other sampling methods I can find in that directions book. 
- From wikipedia: 
  - [Kent distribution](https://en.wikipedia.org/wiki/Kent_distribution) 
  - [von Mises-Fisher](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution#Matrix_Von_Mises-Fisher)



**TODO: add in stuff about the CW complex inputs**
- CW Complexes might be too broad a term. Perhaps "[polygon mesh](https://en.wikipedia.org/wiki/Polygon_mesh)" is better. 

## Distances 

Additional code is included for computing distances between the resulting ECT matrices. 

![MDS of Matisse](figures/Matisse_MDS.png)

## Generalized versions of ECT

The ECT package provides implementations for both the Smooth Euler Characteristic Transform and the Differentiable Euler Characteristic Transform. This allows for users to quickly examine their dataset under the lense of various topological transforms to find what best suits their problem.

# Statement of Need

Despite the ECT's mathematical elegances, there has been a notable absense of efficient, user-friendly Python implementations that can handle the computational demands of modern research datasets. The ECT package addressed this by leveraging Numba's just-in-time compilation to achieve significant speedups over naive Python implementations, making it practical to compute ECTs for large-scale datasets. This performance is then complimented by the many utility functions for visualizing and comparing different Euler Characteristic Tranforms such as the ECT, SECT, and the DECT.

# Representative Publications Using ECT

Have we actually used it yet? 

# Acknowledgements

This material is based in part upon work supported in part by the National Science Foundation through grants
CCF-1907591,
CCF-2106578,
and CCF-2142713.
**Get the rest?**

# References
