.. PantheonRL documentation master file, created by
   sphinx-quickstart on Mon Oct  2 15:09:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PantheonRL Docs
===============

`PantheonRL <https://github.com/Stanford-ILIAD/PantheonRL>`_ is a package for training and testing multi-agent reinforcement learning environments. The goal of PantheonRL is to provide a modular and extensible framework for training agent policies, fine-tuning agent policies, ad-hoc pairing of agents, and more.

PantheonRL is built to support Stable-Baselines3 (SB3), allowing direct access to many of SB3's standard RL training algorithms such as PPO. PantheonRL currently follows a decentralized training paradigm -- each agent is equipped with its own replay buffer and update algorithm. The agents objects are designed to be easily manipulable. They can be saved, loaded and plugged into different training procedures such as self-play, ad-hoc / cross-play, round-robin training, or finetuning.

This package was presented as a demo at the AAAI-22 Demonstrations Program [Sarkar2022]_.

Github repository: https://github.com/Stanford-ILIAD/PantheonRL

Paper: https://aaai.org/papers/13221-pantheonrl-a-marl-library-for-dynamic-training-interactions/

Video: https://youtu.be/3-Pf3zh_Hpo

Citation
--------
::
   
   @inproceedings{sarkar2022pantheonRL,
     title={PantheonRL: A MARL Library for Dynamic Training Interactions},
     author={Sarkar, Bidipta and Talati, Aditi and Shih, Andy and Sadigh Dorsa},
     booktitle = {Proceedings of the 36th AAAI Conference on Artificial Intelligence (Demo Track)},
     year={2022}
   }

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install


.. toctree::
   :maxdepth: 2
   :caption: API reference

   _autosummary/pantheonrl

.. [Sarkar2022] "PantheonRL: A MARL Library for Dynamic Training Interactions"
   Bidipta Sarkar*, Aditi Talati*, Andy Shih*, Dorsa Sadigh
   In Proceedings of the 36th AAAI Conference on Artificial Intelligence (Demo Track), 2022
