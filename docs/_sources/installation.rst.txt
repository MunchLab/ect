.. _installation:

Installation
=============

Prerequisites
-------------

Before installing `ect`, make sure you have the following prerequisites:

- Python (version X.X or higher)
- Pip (Python package installer)

Installing `ect`
-----------------

To install `ect`, follow these steps:

1. Open a terminal or command prompt.

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv myenv
    ```

3. Activate the virtual environment:

    - On Windows:

      ```bash
      myenv\Scripts\activate
      ```

    - On macOS and Linux:

      ```bash
      source myenv/bin/activate
      ```

4. Install `ect` using pip:
   *Please note that this is not yet available on PyPi, so you will need to install it from the source code.*

    
    ```bash
    git clone https://github.com/MunchLab/ect
    ```

    ```bash
    cd ect
    ```

    ```bash
    pip install .
    ```


5. You're all set! You can now start using `ect` in your projects.

Uninstalling `ect`
------------------

To uninstall `ect`, simply run the following command:

    ```bash
    pip uninstall ect
    ```