# `ect`: A python package for computing the Euler Characteristic Transform

Python computation tools for computing the Euler Characteristic Transform of embedded graphs. 

## Description

Right now, the content includes stuff for doing ECT on graphs embedded in 2D. Eventually the goal is to get voxel versions, higher dimensional simplicial complexes, etc in here.

For more information on the ECT, see:

>  Munch, Elizabeth. An Invitation to the Euler Characteristic Transform. The American Mathematical Monthly, 132(1), 15-25. [doi:10.1080/00029890.2024.2409616](https://doi.org/10.1080/00029890.2024.2409616). 2024.

## Getting Started

### Documentation and tutorials

- The documentation is available at: [munchlab.github.io/ect](https://munchlab.github.io/ect/)
- A tutorial jupyter notebook can be found [here](https://munchlab.github.io/ect/notebooks/Tutorial-ECT_for_embedded_graphs.html)

### Dependencies

- `networkx`
- `numpy`
- `matplotlib`
- `numba`

### Installing

The package can be installed using pip:

```{bash}
pip install ect
```

Alternatively, you can clone the repo and install directly

```{bash}
git clone git@github.com:MunchLab/ect.git
cd ect
pip install .
```

## Authors

This code was written by [Liz Munch](https://elizabethmunch.com/) along with her research group and collaborators. People who have contributed to `ect` include:

- [Sarah McGuire](https://www.sarah-mcguire.com/)
- [Yemeen Ayub](https://yemeen.com/)

## License

This project is licensed under the GPLv3 License - see the License file for details

## Contact Information

- Liz Munch: [Website](http://www.elizabethmunch.com), [Email](mailto:muncheli@msu.edu)
