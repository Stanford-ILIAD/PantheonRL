Installation
============

Prerequisites
-------------

We recommend using a virtual environment with python 3.8+, following the same prerequisites as Stable-Baselines3


.. code-block:: console

   (base) $ conda create -n PantheonRL python=3.10
   (base) $ conda activate PantheonRL


Development version
-------------------

Use this version to get the latest features and contribute to PantheonRL. Note that this version is the only one that has the overcooked extension.

.. code-block:: console

   (PantheonRL) $ git clone https://github.com/Stanford-ILIAD/PantheonRL.git
   (PantheonRL) $ cd PantheonRL
   (PantheonRL) $ pip install -e .


Overcooked Installation
^^^^^^^^^^^^^^^^^^^^^^^

To install the overcooked extension (from within the PantheonRL directory):

.. code-block:: console

   (PantheonRL) $ git submodule update --init --recursive
   (PantheonRL) $ pip install -e overcooked_extension


PettingZoo Installation
-----------------------

You can optionally install the pettingzoo environments

.. code-block:: console

   (PantheonRL) $ pip install pettingzoo

You can also install the dependencies for a group of pettingzoo environments. Refer to the `PettingZoo docs <https://pettingzoo.farama.org/>`_ for more information.

.. code-block:: console

   (PantheonRL) $ pip install "pettingzoo[classic]"
