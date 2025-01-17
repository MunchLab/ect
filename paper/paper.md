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

The field of Topological Data Analysis encodes the shape of data in quantifiable representations of the information, sometimes called "topological signatures" or "topological summaries". 
The goal is to ensure that these summaries are robust enough to be useable in the face of noise, while having access to fast enough algorithms to make them useful in practice. 
Often, these two goals are at odds with each other since the more complex the representation in order to retain as much information as possible generally results in larger computation time.  
The Euler Characteristic Transform (ECT) is a construction which is rapidly gaining popularity in Topological Data Analysis settings due to its ability to be both at once: robust to the input providing a provably complete representation of an input embedded shape, while being much faster to compute than its commonly used cousin, the Persistent Homology Transform. 
The *ECT* package for Python presented here provides a fast and well-documented implementation of the ECT for graphs embedded in 2 or 3 dimensions. This new package is particularly timely since access to easy-to-use code will make the ECT accessible to more practitioners and domain scientists interested in using it for applications. ***TODO Assuming we actually have this up to 3d***

# The ECT

**TODO: Get permission for the Matisse example image**

We start by defining the Euler Characteristic Transform, and direct the reader to [@Munch2025] for a full survey article on the subject. 

To start, we assume our input is an undirected graph $G$ with an embedding in 2D given by a map on the vertices $f: V(G) \to \mathbb{R}^2$. A graph can be constructed as follows. 

![An example of an embedded graph](figures/example_graph.png)

For a choice of direction $\theta \in [0,2\pi]$, we can induce a function on the vertex set. 
Thinking of  this as $\omega \in \mathbb{S}^1$ by defining the unit vector $\omega = (\cos(\theta), \sin(\theta))$, the function $g_\omega$ is defined on the vertices of $G$ by taking the dot product of the embedding coordinates with the unit vector, specifically
$$
g_\omega(v) = \langle f(v), \omega\rangle.
$$
<!-- This is done in the code using the `g_omega` method as shown.  -->

Some examples are shown below. 

![Example 1](figures/example_graph_pi_over_2.png)
![Example 2](figures/example_graph_pi_over_2.png)

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
& (\omega,a) & \mapsto & \chi(g_\omega^{-1}(-\infty,a]) 
\end{matrix}
$$
After discretizing, the example embedded graph has an ECT matrix as shown below. 
![Example ECT](figures/example_ect.png)

The main functionality of the package is to be able to compute the ECT matrix for graphs embedded in $\mathbb{R}^d$ for $d \in \{2,3\}$.
Additional functionality is given for embedded CW complexes, such as the example shown below.  

## Extension to higher dimensional embedding

**TODO: If we're going to have 3D stuff implemented, we need the version for that descibed.**

## Extension to higher dimensional cells

**TODO: add in stuff about the CW complex inputs**

## Distances 

Additional code is included for computing distances between the resulting ECT matrices. 

![MDS of Matisse](figures/Matisse_MDS.png)

## Generalized versions of ECT

SECT, DECT, whatever else? 

# Statement of need

**TODO: Are there any existing ECT packages?**

Yemeen is going to find any needed references.

- DECT 
- Demeter 
- 

# Representative Publications Using ECT

Have we actually used it yet? 

# Acknowledgements

This material is based in part upon work supported in part by the National Science Foundation through grants
CCF-1907591,
CCF-2106578,
and CCF-2142713.
**Get the rest?**

# References
