************
Installation
************

Via Python Package
==================

Install the package and required dependencies using:

.. code:: bash

    pip install sinaps

Installation troubleshooting
============================

The installation may fail due to conflicts in the dependencies, or due to old version of already installed librairies.
First you can try to upgrade pip, to use the newest method to resolve dependency conflicts:

.. code:: bash

    pip install pip --upgrade
    
If you cannot upgrade pip, you may try:

.. code:: bash

    pip install sinaps --use-feature=2020-resolver

If it conflicts with other dependencies requirement of your own environment, you could try to install sinaps in a virtual environment:

.. code:: bash

    python3 -m venv sinaps_env
    source sinaps_env/bin/activate
    pip install sinaps
