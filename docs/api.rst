###################
PGMax API reference
###################

This page contains the list of project's modules

.. currentmodule:: pgmax

Factor
===================
.. automodule:: pgmax.factor

.. autosummary::
   factor
   enum
   logical
   pool

factor
~~~~~~~~~~~~~

.. autoclass:: Wiring
.. autoclass:: Factor

enum
~~~~~~~~~~~~~

.. autoclass:: EnumWiring
.. autoclass:: EnumFactor

logical
~~~~~~~~~~~~~

.. autoclass:: LogicalWiring
.. autoclass:: LogicalFactor
.. autoclass:: ORFactor
.. autoclass:: ANDFactor

pool
~~~~~~~~~~~~~

.. autoclass:: PoolWiring
.. autoclass:: PoolFactor


Fgraph
===================
.. automodule:: pgmax.fgraph

.. autosummary::
   fgraph

fgraph
~~~~~~~~~~~~~

.. autoclass:: FactorGraphState
.. autoclass:: FactorGraph


Fgroup
===================
.. automodule:: pgmax.fgroup

.. autosummary::
   enum
   fgroup
   logical
   pool

fgroup
~~~~~~~~~~~~~

.. autoclass:: FactorGroup
.. autoclass:: SingleFactorGroup

enum
~~~~~~~~~~~~~

.. autoclass:: EnumFactorGroup
.. autoclass:: PairwiseFactorGroup

logical
~~~~~~~~~~~~~

.. autoclass:: LogicalFactorGroup
.. autoclass:: ORFactorGroup
.. autoclass:: ANDFactorGroup

pool
~~~~~~~~~~~~~

.. autoclass:: PoolFactorGroup
   


Infer
===================
.. automodule:: pgmax.infer

.. autosummary::
   bp
   bp_state
   energy

bp
~~~~~~~~~~~~~

.. autoclass:: BeliefPropagation
.. autofunction:: BP
.. autofunction:: decode_map_states
.. autofunction:: get_marginals

bp_state
~~~~~~~~~~~~~

.. autoclass:: BPArrays
.. autoclass:: LogPotentials
.. autoclass:: FToVMessages
.. autoclass:: Evidence
.. autoclass:: BPState

energy
~~~~~~~~~~~~~

.. autofunction:: compute_energy


Vgroup
===================
.. automodule:: pgmax.vgroup

.. autosummary::
   vgroup
   varray
   vdict

vgroup
~~~~~~~~~~~~~

.. autoclass:: VarGroup

varray
~~~~~~~~~~~~~

.. autoclass:: NDVarArray

vdict
~~~~~~~~~~~~~

.. autoclass:: VarDict


Utils
===================
.. automodule:: pgmax.utils