ect: Euler Characteristic Transform in Python
=============================================

Python computation tools for computing the Euler Characteristic Transform of embedded graphs. 

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :numbered:
   :caption: Contents:

   Getting Started <installation.rst>
   Modules <modules.rst>
   Tutorials <tutorials.md>
   Contributing <contributing.rst>
   License <license.md>
   Citing <citing.rst>

Description
-----------

Right now, the content includes stuff for doing ECT on graphs embedded in 2D. Eventually the goal is to get voxel versions, higher dimensional simplicial complexes, etc in here.

For more information on the ECT, see:

   Elizabeth Munch. An Invitation to the Euler Characteristic Transform. `arXiv:2310.10395 <https://arxiv.org/abs/2310.10395>`_. 2023.

Getting Started
---------------

Documentation and tutorials
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The documentation is available at: `munchlab.github.io/ect <https://munchlab.github.io/ect/>`_
* A tutorial jupyter notebook can be found `here <https://munchlab.github.io/ect/notebooks/Tutorial-ECT_for_embedded_graphs.html>`_
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

License
-------

This project is licensed under the GPLv3 License - see the License file for details

Contact Information
-------------------

- Liz Munch: `Website <http://www.elizabethmunch.com>`_, `Email <mailto:muncheli@msu.edu>`_
