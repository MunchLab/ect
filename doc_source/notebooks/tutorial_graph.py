# %% [markdown]
# # Tutorial : ECT for embedded graphs
#
# This jupyter notebook will walk you through using the `ect` package to compute the Euler characteristic transform of a 2D embedded graph. This tutorial assumes you already know what an ECT is; see [this paper](https://arxiv.org/abs/2310.10395) for a more thorough treatment of details.

# %%
from ect import ECT, EmbeddedGraph
from ect.utils.examples import create_example_graph

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# %% [markdown]
# ## Constructing the embedded graph
#
# We assume our input is an undirected graph $G$ with an embedding in 2D given by a map on the vertices $f: V(G) \to \mathbb{R}^2$. A graph can be constructed as follows.
#

# %%
# Construct an example graph
# Note that this is the same graph that is returned by:
# G = create_example_graph()

G = EmbeddedGraph()

G.add_node("A", [1, 2])
G.add_node("B", [3, 4])
G.add_node("C", [5, 7])
G.add_node("D", [3, 6])
G.add_node("E", [4, 3])
G.add_node("F", [4, 5])

G.add_edge("A", "B")
G.add_edge("B", "C")
G.add_edge("B", "D")
G.add_edge("B", "E")
G.add_edge("C", "D")
G.add_edge("E", "F")

G.plot()


# %% [markdown]
# The coordinates of all vertices, given as a dictionary, can be accessed using the `coord_matrix` attribute.

# %%
G.coord_matrix

# %% [markdown]
# Because of the rotational aspect of the ECT, we often want our graph to be centered, so you can use the `center_coordinates` method shift the graph to have the average of the vertex coordinates be 0. Note that this does overwrite the coordinates of the points.

# %%
G.center_coordinates(center_type="mean")
print(G.coord_matrix)
G.plot()

# %% [markdown]
# To get a bounding radius we can use the `get_bounding_radius` method.

# %%
# This is actually getting the radius
r = G.get_bounding_radius()
print(f"The radius of bounding circle centered at the origin is {r}")

# plotting the graph with it's bounding circle of radius r.
G.plot(bounding_circle=True)


# %% [markdown]
# We can also rescale our graph to have unit radius using `scale_coordinates`

# %%
G.scale_coordinates(radius=1)
G.plot(bounding_circle=True)

r = G.get_bounding_radius()
print(f"The radius of bounding circle centered at the origin is {r}")


# %%
myect = ECT(num_dirs=16, num_thresh=20)

# The ECT object will automatically create directions when needed
print(f"Number of directions: {myect.num_dirs}")
print(f"Number of thresholds: {myect.num_thresh}")

# %% [markdown]
# We can set the bounding radius as follows. Note that some methods will automatically use the bounding radius of the input `G` if not already set. I'm choosing the radius to be a bit bigger than the bounding radius of `G` to make some better pictures.

# %%
myect.set_bounding_radius(1.2 * G.get_bounding_radius())

print(f"Internally set radius is: {myect.bound_radius}")
print(f"Thresholds chosen are: {myect.thresholds}")

# %% [markdown]
# If we want the Euler characteristic curve for a fixed direction, we use the `calculate` function with a specific angle. This returns an ECTResult object containing the computed values.

# %%
result = myect.calculate(G, theta=np.pi / 2)
print(f"ECT values for direction pi/2: {result[0]}")

# %% [markdown]
# To calculate the full ECT, we call the `calculate` method without specifying theta. The result returns the ECT matrix and associated metadata.

# %%
result = myect.calculate(G)

print(f"ECT matrix shape: {result.shape}")
print(f"Number of directions: {myect.num_dirs}")
print(f"Number of thresholds: {myect.num_thresh}")

# We can plot the result matrix
result.plot()

# %% [markdown]
# ## SECT
#
# The Smooth Euler Characteristic Transform (SECT) can be calculated from the ECT. Fix a radius $R$ bounding the graph. The average ECT in a direction $\omega$ defined on function values $[-R,R]$ is given by
# $$\overline{\text{ECT}_\omega} = \frac{1}{2R} \int_{t = -R}^{R} \chi(g_\omega^{-1}(-\infty,t]) \; dt. $$
# Then the SECT is defined by
# $$
# \begin{matrix}
# \text{SECT}(G): & \mathbb{S}^1 & \to & \text{Func}(\mathbb{R}, \mathbb{Z})\\
# & \omega & \mapsto & \{ t \mapsto \int_{-R}^t \left( \chi(g_\omega^{-1}(-\infty,a]) -\overline{\text{ECT}_\omega}\right)\:da \}
# \end{matrix}
# $$

# %% [markdown]
# The SECT can be computed from the ECT result:

# %%
sect = result.smooth()

sect.plot()
