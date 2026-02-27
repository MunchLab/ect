ect: Euler Characteristic Transform in Python
=============================================

The `ect` package is a library of tools for computing the Euler Characteristic Transform of embedded cell complexes with arbitrary dimensional cells. This package is to aid researchers in quickly extracting topological information from their data via the Euler Characteristic Transform. 

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :numbered:
   :caption: Contents:

   Getting Started <installation.rst>
   Modules <modules.rst>
   Tutorials <tutorials.rst>
   Contributing <contributing.rst>
   License <license.md>
   Citing <citing.rst>

Description
-----------

This package provides tools for computing the Euler Characteristic Transform (ECT) of embedded cell complexes efficienctly and provides many useful utilities for visualizing and comparing different ECTs.

For more information on the ECT, see:

   Munch, Elizabeth. An Invitation to the Euler Characteristic Transform. The American Mathematical Monthly, 132(1), 15-25. `doi:10.1080/00029890.2024.2409616 <https://doi.org/10.1080/00029890.2024.2409616>`_. 2024.

Getting Started
---------------

Documentation and tutorials
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The documentation is available at: `munchlab.github.io/ect <https://munchlab.github.io/ect/>`_
* A comprehensive tutorial for the unified `EmbeddedComplex` class can be found `here <https://munchlab.github.io/ect/notebooks/Tutorial-EmbeddedComplex.html>`_
* The source code can be found at: `github.com/MunchLab/ect  <https://github.com/MunchLab/ect>`_

Dependencies
^^^^^^^^^^^^

* `networkx`
* `numpy`
* `matplotlib`
* `numba`

Installing
^^^^^^^^^^

The package can be installed using pip:

.. code-block:: bash

   pip install ect


Alternatively, you can clone the repo and install directly

.. code-block:: bash

   git clone git@github.com:MunchLab/ect.git
   cd ect
   pip install .

Authors
-------

This code was written by `Liz Munch <https://elizabethmunch.com/>`_ along with her research group and collaborators. People who have contributed to `ect` include:

- `Sarah McGuire <https://www.sarah-mcguire.com/>`_
- `Yemeen Ayub <https://yemeen.com/>`_

License
-------

This project is licensed under the GPLv3 License - see the License file for details

Contact Information
-------------------

- Liz Munch: `Website <http://www.elizabethmunch.com>`_, `Email <mailto:muncheli@msu.edu>`_
