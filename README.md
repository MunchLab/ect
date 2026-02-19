# `ect`: A python package for computing the Euler Characteristic Transform

Python computation tools for computing the Euler Characteristic Transform of embedded complexes. 

## Description

The package provides fast tools for computing the Euler Characteristic Transform (ECT) on embedded cell complexes in any ambient dimension. You build a complex (vertices, edges, and optional higher‑dimensional cells), choose a set of directions and thresholds, and compute either the exact ECT or its smoothed/differentiable variants. Results come back as NumPy arrays with metadata, plotting helpers, and distance utilities, making it straightforward to visualize transforms and compare shapes. The core is implemented with NumPy and Numba, with optional validation of geometric and structural constraints when constructing complexes.

- `EmbeddedComplex`: convert point clouds into complexes with vertices, edges, and higher‑dimensional cells with embedded coordinates.
- `ECT`, `SECT`, `DECT` : ECT calculations along with the smooth and differentiable variants over sampled directions and transforms.
- `Directions`: uniform, random, or custom directions (angles in 2D; vectors in any dimension)
- Results as `ECTResult`: behaves like a NumPy array, with plotting and distance helpers
- Optional geometric/structural validation when building complexes

For more information on the ECT, see:

>  Munch, Elizabeth. An Invitation to the Euler Characteristic Transform. The American Mathematical Monthly, 132(1), 15-25. [doi:10.1080/00029890.2024.2409616](https://doi.org/10.1080/00029890.2024.2409616). 2024.

## Getting Started

### Documentation and tutorials

- The documentation is available at: [munchlab.github.io/ect](https://munchlab.github.io/ect/)
- A tutorial jupyter notebook can be found [here](https://munchlab.github.io/ect/notebooks/Tutorial-EmbeddedComplex.html)


### Installing

Requires Python 3.10+.

Install from PyPI:

```{bash}
pip install ect
```

Alternatively, you can clone the repo and install directly

```{bash}
git clone git@github.com:MunchLab/ect.git
cd ect
pip install .
```

### Quickstart

Compute an ECT for a simple embedded triangle and plot it.

```python
from ect import ECT, EmbeddedComplex

G = EmbeddedComplex()
G.add_node("a", [0.0, 0.0])
G.add_node("b", [1.0, 0.0])
G.add_node("c", [0.5, 0.8])
G.add_edge("a", "b")
G.add_edge("b", "c")
G.add_edge("c", "a")

ect = ECT(num_dirs=32, num_thresh=128)
result = ect.calculate(G)
result.plot()
```



## Authors

This code was written by [Liz Munch](https://elizabethmunch.com/) along with her research group and collaborators. People who have contributed to `ect` include:

- [Sarah McGuire](https://www.sarah-mcguire.com/)
- [Yemeen Ayub](https://yemeen.com/)
- [Dan Chitwood](https://github.com/DanChitwood)

## License

This project is licensed under the GPLv3 License - see the License file for details

## Contact Information

- Liz Munch: [Website](http://www.elizabethmunch.com), [Email](mailto:muncheli@msu.edu)
